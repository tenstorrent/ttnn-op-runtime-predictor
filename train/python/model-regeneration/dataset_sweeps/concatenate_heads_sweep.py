# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import itertools

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes, sanitize_shape_rm
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

batch_sizes = [1, 32, 64, 96, 128]
num_heads = [16, 24, 32, 40, 48, 52, 64, 80, 96, 128]
seq_lens = [1, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
head_dims = [64, 128]

shapes = [list(shape) for shape in itertools.product(batch_sizes, num_heads, seq_lens, head_dims)]
#split shapes based on seq_len
shapes_64 = [shape for shape in shapes if shape[2] <= 64]
shapes_128 = [shape for shape in shapes if shape[2] > 64 and shape[2] <= 128]
shapes_256 = [shape for shape in shapes if shape[2] > 128 and shape[2] <= 256]
shapes_512 = [shape for shape in shapes if shape[2] > 256 and shape[2] <= 512]
shapes_1024 = [shape for shape in shapes if shape[2] > 512 and shape[2] <= 1024]
shapes_2048 = [shape for shape in shapes if shape[2] > 1024 and shape[2] <= 2048]
shapes_4096 = [shape for shape in shapes if shape[2] > 2048 and shape[2] <= 4096]
shapes_8192 = [shape for shape in shapes if shape[2] > 4096 and shape[2] <= 8192]
shapes_16384 = [shape for shape in shapes if shape[2] > 8192 and shape[2] <= 16384]
shapes_32768 = [shape for shape in shapes if shape[2] > 16384 and shape[2] <= 32768]
shapes_65536 = [shape for shape in shapes if shape[2] > 32768 and shape[2] <= 65536]
shapes_131072 = [shape for shape in shapes if shape[2] > 65536 and shape[2] <= 131072]

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "suite_64": {
        "input_shape": shapes_64,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "suite_128": {
        "input_shape": shapes_128,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "suite_256": {
        "input_shape": shapes_256,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "suite_512": {
        "input_shape": shapes_512,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "suite_1024": {
        "input_shape": shapes_1024,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "suite_2048": {
        "input_shape": shapes_2048,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "suite_4096": {
        "input_shape": shapes_4096,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "suite_8192": {
        "input_shape": shapes_8192,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "suite_16384": {
        "input_shape": shapes_16384,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "suite_32768": {
        "input_shape": shapes_32768,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "suite_65536": {
        "input_shape": shapes_65536,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "suite_131072": {
        "input_shape": shapes_131072,
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}

def get_volume(test_vector):
    shape = test_vector["input_shape"]
    volume = 1
    for dim in shape:
        volume *= dim
    return volume

def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_memory_config"] == ttnn.L1_MEMORY_CONFIG and test_vector["output_memory_config"] == ttnn.L1_MEMORY_CONFIG:
        if test_vector["input_dtype"] == ttnn.bfloat16:
            if get_volume(test_vector) >= 17825792:
                return True, "Input shape too large for L1 to L1 concat heads"
        if test_vector["input_dtype"] == ttnn.bfloat8_b:
            if get_volume(test_vector) >= 35651584:
                return True, "Input shape too large for L1 to L1 concat heads"
    if (test_vector["input_memory_config"] == ttnn.L1_MEMORY_CONFIG and test_vector["output_memory_config"] == ttnn.DRAM_MEMORY_CONFIG) or (test_vector["input_memory_config"] == ttnn.DRAM_MEMORY_CONFIG and test_vector["output_memory_config"] == ttnn.L1_MEMORY_CONFIG):
        if test_vector["input_dtype"] == ttnn.bfloat16:
            if get_volume(test_vector) >= 29360128:
                return True, "Input shape too large for L1 to DRAM or DRAM to L1 concat heads"
        if test_vector["input_dtype"] == ttnn.bfloat8_b:
            if get_volume(test_vector) >= 58720256:
                return True, "Input shape too large for L1 to DRAM or DRAM to L1 concat heads"
    if test_vector["input_memory_config"] == ttnn.DRAM_MEMORY_CONFIG and test_vector["output_memory_config"] == ttnn.DRAM_MEMORY_CONFIG:
        if test_vector["input_dtype"] == ttnn.bfloat16:
            if get_volume(test_vector) >= 88080384:
                return True, "Input shape too large for DRAM to DRAM concat heads"
        if test_vector["input_dtype"] == ttnn.bfloat8_b:
            if get_volume(test_vector) >= 176160768:
                return True, "Input shape too large for DRAM to DRAM concat heads"
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
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.transformer.concatenate_heads(input_tensor, memory_config=output_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    return [[True, ""], e2e_perf]
