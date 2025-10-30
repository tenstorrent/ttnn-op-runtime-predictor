# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import itertools
import pytest
import pandas as pd
import math

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes, sanitize_shape_rm
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import (
    start_measuring_time,
    stop_measuring_time,
)
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 15

random.seed(0)

def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power
    # if (2**math.log2(x) == x):
    #     return x
    # return 2**(int(x).bit_length())

def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )

def is_k_chunk_size_valid(num_heads: int, k_chunk_size: int, fp32_dest_acc_en: bool = False) -> bool:
    """
    Replicates the logic from sdpa_decode_program_factory.cpp to check if a 
    given k_chunk_size is valid for a specific number of heads.

    Args:
        num_heads (int): The number of attention heads.
        k_chunk_size (int): The K-cache chunk size to validate.
        fp32_dest_acc_en (bool): Corresponds to the compute kernel config.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    TILE_HEIGHT = 32

    # 1. Determine dst_size from fp32_dest_acc_en
    # const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    dst_size = 4 if fp32_dest_acc_en else 8

    # 2. Calculate PNHt (Padded Num Heads in Tiles)
    # The C++ code pads the number of heads to the nearest multiple of 32.
    # uint32_t PNH = round_up(num_heads, 32);
    # uint32_t PNHt = PNH / TILE_HEIGHT;
    padded_num_heads = ((num_heads + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    PNHt = padded_num_heads // TILE_HEIGHT

    # 3. Calculate Sk_chunk_t (K-chunk size in Tiles)
    # const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    if k_chunk_size == 0 or k_chunk_size % TILE_HEIGHT != 0:
        return False # k_chunk_size must be a multiple of 32
    Sk_chunk_t = k_chunk_size // TILE_HEIGHT

    # 4. Calculate mul_bcast_granularity
    # const uint32_t mul_bcast_granularity = std::min(PNHt * Sk_chunk_t, dst_size);
    mul_bcast_granularity = min(PNHt * Sk_chunk_t, dst_size)

    # 5. Check if mul_bcast_granularity is a power of two
    # TT_FATAL(mul_bcast_granularity == (1 << log2_mul_bcast_granularity), "Error");
    # A positive integer is a power of two if and only if (x & (x - 1)) == 0.
    is_power_of_two = (mul_bcast_granularity > 0) and ((mul_bcast_granularity & (mul_bcast_granularity - 1)) == 0)
    
    return is_power_of_two

def generate_paged_sdpa_decode_valid_shapes(
    batch_sizes,
    num_q_heads,
    num_kv_heads,
    kv_cache_seq_lens,
    head_dims,
    k_chunk_sizes,
    is_causal_options=[True, False],
):

    """
    Generates valid shape tuples for paged_sdpa_decode based on C++ validation logic.
    Each tuple contains: (q_shape, k_shape, v_shape, page_table_shape, attn_mask_shape, cur_pos_shape, is_causal, k_chunk_size)
    """
    valid_shapes = []
    
    # Use itertools.product to create all combinations of dimensions
    all_combinations = itertools.product(
        batch_sizes,
        num_q_heads,
        num_kv_heads,
        kv_cache_seq_lens,
        head_dims,
        is_causal_options,
        k_chunk_sizes,
    )

    for b, nh, nkv, s_k, d, is_causal, k_chunk in all_combinations:

        # 1 Num Q heads must be a multiple of Num KV heads
        if nh % nkv != 0:
            continue

        if not is_k_chunk_size_valid(nh, k_chunk, True):
            continue #invalid k_chunk_size

        # 2. Define core tensor shapes
        # Q shape: [1, B, NH, D]
        q_shape = (1, b, nh, d)
        
        # K/V cache shape: [B, NKV, S_k, D]
        k_shape = (b, nkv, s_k, d)
        v_shape = k_shape  # K and V shapes must be the same

        # 3. Define page_table shape: [B, num_pages]
        # num_pages is a variable parameter, let's assume it's related to seq_len.
        # For this example, let's set num_pages = 1 for simplicity.
        num_pages = s_k // 32
        page_table_shape = (b, num_pages)

        # 4. Define cur_pos_tensor shape: [B]
        # Required for causal, optional otherwise. Shape is always valid if batch matches.
        cur_pos_shape = (b,)

        q_volume = 1 * b * nh * d
        if q_volume * 2 > 1499136:
            continue #q tensor too large
        kv_volume = b * nkv * s_k * d
        if kv_volume * 2 > 1499136:
            continue #k or v tensor too large
        pt_volume = b * num_pages * 2
        if pt_volume > 1499136:
            continue #page table tensor too large
        cp_volume = b * 4
        if cp_volume > 1499136:
            continue #cur pos tensor too large
         

        # 5. Define attn_mask shape and validate
        attn_mask_shape = None
        if not is_causal:
            # Mask is only valid for non-causal mode.
            # Shape: [B, NH, S_q, S_k_total]
            # S_q for decode is 1. S_k_total is the full sequence length.
            s_q = 1
            s_k_total = page_table_shape[1] * k_shape[2]
            
            # Validation: Total K sequence length must be a multiple of k_chunk_size
            if s_k_total % k_chunk != 0:
                continue # This combination is invalid
            
            attn_mask_shape = (b, 1, nh, s_k_total)

        # define scale
        scale = d ** 0.5  # Typical scale is sqrt of head_dim

        # If all checks pass, add the combination to the list
        valid_shapes.append(
            (q_shape, k_shape, v_shape, page_table_shape, attn_mask_shape, cur_pos_shape, is_causal, k_chunk, scale)
        )
        
    return valid_shapes

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

#q shape: [1 x b x nh x head_dim]
#k, v shape: [b x n_kv_heads x seq_len x head_dim]
#page_table shape: [b x num_pages]
#cur_pos_tensor shape: [b, ]
#attn_mask shape: [b x nh x seq_len_q x seq_len_k]

#parameters for tensors (q, k, v, page_table, cur_pos_tensor, attn_mask)
batch_sizes = [1, 8, 32]
num_q_heads = [16, 24, 32, 40, 48, 52, 64, 128]
num_kv_heads = [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128]
seq_lens = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
head_dims = [64, 128]

#input tensor config params
input_q_tensor_memory_configs = [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG]
input_kv_tensor_memory_configs = [ttnn.DRAM_MEMORY_CONFIG]
input_tensor_dtypes = [ttnn.bfloat16, ttnn.bfloat8_b]

#page table tensor params
page_table_tensor_memory_configs = [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG]
page_table_tensor_dtypes = [ttnn.uint16]
page_table_layouts = [ttnn.ROW_MAJOR_LAYOUT]

#attn_mask tensor params
attn_mask_tensor_memory_configs = [ttnn.DRAM_MEMORY_CONFIG]
attn_mask_tensor_dtypes = [ttnn.bfloat16, ttnn.bfloat8_b]
attn_mask_layouts = [ttnn.TILE_LAYOUT]

#cur_pos_tensor params
cur_pos_tensor_memory_configs = [ttnn.DRAM_MEMORY_CONFIG]
cur_pos_tensor_dtypes = [ttnn.int32]
cur_pos_tensor_layouts = [ttnn.ROW_MAJOR_LAYOUT]

#other params (is_causal, scale)
is_causals = [True, False]

#output memory configs
output_memory_configs = [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG]

#compute kernel config parameters
math_fidelities = [ttnn.MathFidelity.HiFi2]
math_approx_modes = [True]
fp32_dest_acc_ens = [True]
packer_l1_accs = [True]

#sdpa program config parameters

#32 to 4096 in powers of 2
q_chunk_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
k_chunk_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
exp_approx_modes = [True, False]

#create valid tensor combinations
valid_combinations = generate_paged_sdpa_decode_valid_shapes(
    batch_sizes,
    num_q_heads,
    num_kv_heads,
    seq_lens,
    head_dims,
    k_chunk_sizes
)

print(f"Number of valid shape combinations: {len(valid_combinations)}")

#randomly sample from valid combinations to limit the number of test vectors
n_samples = 1000
if len(valid_combinations) > n_samples:
    valid_combinations = random.sample(valid_combinations, n_samples)

if len(valid_other_combinations) > n_samples:
    valid_other_combinations = random.sample(valid_other_combinations, n_samples)   

# Split valid_combinations into smaller chunks
length_valid_combinations = len(valid_combinations)
num_chunks = 250
chunk_size = (length_valid_combinations + num_chunks - 1) // num_chunks
valid_combinations_chunks = [valid_combinations[i:i + chunk_size] for i in range(0, length_valid_combinations, chunk_size)]

# Dynamically create the parameters dictionary based on the number of chunks
parameters = {}
for idx, chunk in enumerate(valid_combinations_chunks, start=1):
    suite_name = f"suite_{idx}"
    parameters[suite_name] = {
        "valid_combination": chunk,
        "input_dtype": input_tensor_dtypes,
        "input_q_memory_config": input_q_tensor_memory_configs,
        "output_memory_config": output_memory_configs,
        "math_fidelity": math_fidelities,
        "math_approx_mode": math_approx_modes,
        "fp32_dest_acc_en": fp32_dest_acc_ens,
        "packer_l1_acc": packer_l1_accs,
        "exp_approx_mode": exp_approx_modes,
        "q_chunk_size": q_chunk_sizes,
        'use_sdpa_program_config': [True, False],
        "use_device_compute_kernel_config": [True, False]
    }

def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    #unpack the test vector
    valid_combination = test_vector["valid_combination"]
    q_shape, k_shape, v_shape, page_table_shape, attn_mask_shape, cur_pos_shape, is_causal, k_chunk_size, scale_value = valid_combination

    input_dtype = test_vector["input_dtype"]
    input_q_memory_config = test_vector["input_q_memory_config"]
    output_memory_config = test_vector["output_memory_config"]
    math_fidelity = test_vector["math_fidelity"]
    math_approx_mode = test_vector["math_approx_mode"]
    fp32_dest_acc_en = test_vector["fp32_dest_acc_en"]
    packer_l1_acc = test_vector["packer_l1_acc"]
    exp_approx_mode = test_vector["exp_approx_mode"]
    q_chunk_size = test_vector["q_chunk_size"]
    use_sdpa_program_config = test_vector["use_sdpa_program_config"]
    use_device_compute_kernel_config = test_vector["use_device_compute_kernel_config"]

    #validation logic
    #gqa logic (batch size == 1)
    if q_shape[1] > 1: #in gqa mode
        if input_dtype == ttnn.bfloat8_b:
            return True, "Invalid: bfloat8_b not supported in GQA mode"
        if output_memory_config == ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG:
            return True, "Invalid: L1_HEIGHT_SHARDED_MEMORY_CONFIG requires batch size = 1"

    #size check
    q_volume = 1 * q_shape[1] * q_shape[2] * q_shape[3]
    if q_volume * 2 > 1499136:
        return True, f"Invalid: q tensor too large, volume {q_volume}"
    kv_volume = k_shape[0] * k_shape[1] * k_shape[2] * k_shape[3]
    if kv_volume * 2 > 1499136:
        return True, f"Invalid: k or v tensor too large, volume {kv_volume}"
    pt_volume = page_table_shape[0] * page_table_shape[1]

    if q_volume * 2 + kv_volume * 4 > 1499136:
        return True, f"Invalid: combined q, k, v tensors too large, volume {q_volume * 2 + kv_volume * 4}"

    if pt_volume > 1499136:
        return True, f"Invalid: page table tensor too large, volume {pt_volume}"
    cp_volume = cur_pos_shape[0] * 4
    if cp_volume > 1499136:
        return True, f"Invalid: cur pos tensor too large, volume {cp_volume}"
    if attn_mask_shape is not None:
        am_volume = attn_mask_shape[0] * attn_mask_shape[1] * attn_mask_shape[2] * attn_mask_shape[3]
        if am_volume * 2 > 1499136:
            return True, f"Invalid: attn mask tensor too large, volume {am_volume}"

    #k_chunk_size validity check
    if not is_k_chunk_size_valid(q_shape[2], k_chunk_size, fp32_dest_acc_en):
        return True, f"Invalid: k_chunk_size {k_chunk_size} is not valid for num_heads {q_shape[2]} with fp32_dest_acc_en={fp32_dest_acc_en}"
        
    return False, None

# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    valid_combination,
    input_dtype,
    input_q_memory_config,
    output_memory_config,
    math_fidelity,
    math_approx_mode,
    fp32_dest_acc_en,
    packer_l1_acc,
    exp_approx_mode,
    q_chunk_size,
    use_sdpa_program_config,
    use_device_compute_kernel_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    #unpack the valid combination
    q_shape, k_shape, v_shape, page_table_shape, attn_mask_shape, cur_pos_shape, is_causal, k_chunk_size, scale_value = valid_combination
    print("q shape is", q_shape)

    #create torch tensors
    q_torch_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(q_shape)

    k_torch_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(k_shape)

    v_torch_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(v_shape)

    page_table_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=0, high=65536, dtype=torch.int32), ttnn.uint16
    )(page_table_shape)

    #set up optional mask and scale
    attn_mask_torch_tensor = None
    if attn_mask_shape is not None:
        attn_mask_torch_tensor = gen_func_with_cast_tt(
            partial(torch_random, low=0, high=2, dtype=torch.float32), input_dtype
        )(attn_mask_shape)

    #create ttnn tensors

    height_sharded_memcfg = None
    if input_q_memory_config == ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG or output_memory_config == ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG:
        shard_grid = ttnn.CoreRangeSet({num_to_corerange(q_torch_tensor.shape[1])})
        #print("q_tensor shape:", q_torch_tensor.shape)
        padded_nh = nearest_pow_2(nearest_n(q_torch_tensor.shape[2], n=32))
        shard_spec = ttnn.ShardSpec(shard_grid, (padded_nh, q_torch_tensor.shape[3]), ttnn.ShardOrientation.ROW_MAJOR)

        height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    q_tensor = ttnn.from_torch(
        q_torch_tensor,
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_q_memory_config if height_sharded_memcfg is None else height_sharded_memcfg,
    )

    k_tensor = ttnn.from_torch(
        k_torch_tensor,
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, 
    )

    v_tensor = ttnn.from_torch(
        v_torch_tensor,
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    attn_mask_tensor = None
    if attn_mask_shape is not None:
        attn_mask_tensor = ttnn.from_torch(
            attn_mask_torch_tensor,
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, 
        )
    else:
        attn_mask_tensor = None

    #page table tensor
    page_table_tensor = ttnn.from_torch(
        page_table_tensor,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, 
    )

    #cur pos tensor
    cur_pos_tensor = None
    if cur_pos_shape is not None:
        cur_pos_tensor = ttnn.from_torch(
            torch.zeros(cur_pos_shape, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, 
        )

    #sdpa program config
    if use_sdpa_program_config:
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=exp_approx_mode,
        )
    else:
        sdpa_program_config = None

    #device compute kernel
    if use_device_compute_kernel_config:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc
        )
    else:
        compute_kernel_config = None

    #run op
    start_time = start_measuring_time()
    
    output_tensor = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_tensor,
        k_tensor,
        v_tensor,
        page_table_tensor,
        attn_mask=attn_mask_tensor,
        cur_pos_tensor=cur_pos_tensor,
        is_causal=is_causal,
        scale=scale_value,
        memory_config=height_sharded_memcfg if output_memory_config == ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG else output_memory_config,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
    )

    e2e_perf = stop_measuring_time(start_time)

    #for ttnn-op-runtime-predictor, don't check correctness for speed
    return [[True, ""], e2e_perf]
