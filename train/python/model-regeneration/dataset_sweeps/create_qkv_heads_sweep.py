# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import itertools
import pytest

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes, sanitize_shape_rm
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import (
    check_with_pcc,
    get_per_core_size_and_num_cores,
    start_measuring_time,
    stop_measuring_time,
)
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 20

random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

batch_sizes = [1, 2, 4, 6, 8]
dim_2 = [1]
seq_lens = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
head_dims = [64, 128]

shapes = [list(shape) for shape in itertools.product(batch_sizes, dim_2, seq_lens, head_dims)]

#break up shapes list by seq_len, one list per seq_len value

shapes_32 = [shape for shape in shapes if shape[2] == 32]
shapes_64 = [shape for shape in shapes if shape[2] == 64]
shapes_128 = [shape for shape in shapes if shape[2] == 128]
shapes_256 = [shape for shape in shapes if shape[2] == 256]
shapes_512 = [shape for shape in shapes if shape[2] == 512]
shapes_1024 = [shape for shape in shapes if shape[2] == 1024]
shapes_2048 = [shape for shape in shapes if shape[2] == 2048]
shapes_4096 = [shape for shape in shapes if shape[2] == 4096]
shapes_8192 = [shape for shape in shapes if shape[2] == 8192]
shapes_16384 = [shape for shape in shapes if shape[2] == 16384]
shapes_32768 = [shape for shape in shapes if shape[2] == 32768]
shapes_65536 = [shape for shape in shapes if shape[2] == 65536]
shapes_131072 = [shape for shape in shapes if shape[2] == 131072]

parameters = {
    "suite_32": {
        "input_shape": shapes_32,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_64": {
        "input_shape": shapes_64,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_128": {
        "input_shape": shapes_128,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_256": {
        "input_shape": shapes_256,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_512": {
        "input_shape": shapes_512,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_1024": {
        "input_shape": shapes_1024,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_2048": {
        "input_shape": shapes_2048,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_4096": {
        "input_shape": shapes_4096,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_8192": {
        "input_shape": shapes_8192,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_16384": {
        "input_shape": shapes_16384,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_32768": {
        "input_shape": shapes_32768,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_65536": {
        "input_shape": shapes_65536,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
    "suite_131072": {
        "input_shape": shapes_131072,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
        'num_heads': [16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'num_kv_heads': [1, 8, 16, 24, 32, 40, 48, 52, 64, 80, 96, 128],
        'transpose_k_heads': [True, False],
    },
}

def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:

    if test_vector['num_heads'] % test_vector['num_kv_heads'] != 0:
        return True, f"num_heads {test_vector['num_heads']} must be a multiple of num_kv_heads {test_vector['num_kv_heads']}"

    if test_vector['input_shape'][2] %  32 != 0:
        return True, f"seq_len {test_vector['input_shape'][2]} must be a multiple of 32"

    hidden_dim = (test_vector['num_heads'] + 2*test_vector['num_kv_heads']) * test_vector['input_shape'][3]
    if hidden_dim % 32 != 0:
        return True, f"hidden_dim {hidden_dim} must be a multiple of 32"
    return False, None

# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_dtype,
    input_layout,
    input_memory_config,
    output_memory_config,
    num_heads,
    num_kv_heads,
    transpose_k_heads,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    #calculate input tensor shape
    batch_size = input_shape[0]
    seq_len = input_shape[2]
    head_dim = input_shape[3]
    hidden_dim = (num_heads + 2*num_kv_heads) * head_dim
    input_tensor_shape = (batch_size, 1, seq_len, hidden_dim)

    #random torch tensor
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_tensor_shape)

    #determine shard grid
    core_grid = device.compute_with_storage_grid_size()
    num_cores = batch_size  # Start with number of cores equal to batch size
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, core_grid, row_wise=False)
    print("batch_size:", batch_size)

    """ if batch_size <= 8:
        grid_height = batch_size
        grid_width = 1
        shard_grid = ttnn.CoreGrid(y=grid_height, x=grid_width)  # This creates 8x8 grid!
    else:
        grid_height = 8
        grid_width = batch_size // grid_height
        print("grid_height:", grid_height)
        print("grid_width:", grid_width)
        shard_grid = ttnn.CoreGrid(y=grid_height, x=grid_width)  # This creates 8x8 grid! """

    print("shard_grid:", shard_grid)

    #memory config
    input_mem_config = ttnn.create_sharded_memory_config(
        shape=input_tensor_shape,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    #input ttnn tensor
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, # Always load to DRAM first
    )

    #move tensor to sharded memory config
    if input_memory_config.is_sharded():
        input_tensor = ttnn.to_memory_config(input_tensor, memory_config=input_mem_config)

    #run op
    start_time = start_measuring_time()
    output_tensor = ttnn.experimental.create_qkv_heads(
        input_tensor,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k_heads,
        memory_config=output_memory_config
    )
    e2e_perf = stop_measuring_time(start_time)

    #for ttnn-op-runtime-predictor, don't check correctness for speed
    return [[True, ""], e2e_perf]
