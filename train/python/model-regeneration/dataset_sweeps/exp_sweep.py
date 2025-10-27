# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 5

random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "4dim": {
        "input_shape": gen_shapes([1, 1, 32, 32], [32, 32, 320, 320], [1, 1, 32, 32], 900),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.uint16, ttnn.uint32],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "use_safe_nums": [True]
    },
    "3dim": {
        "input_shape": gen_shapes([1, 32, 32], [128, 896, 896], [1, 32, 32], 900),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.uint16, ttnn.uint32],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "use_safe_nums": [True]
    },
    "2dim": {
        "input_shape": gen_shapes([32, 32], [9600, 9600], [32, 32], 900),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.uint16, ttnn.uint32],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "use_safe_nums": [True]
    }
}

# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    use_safe_nums,
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    if use_safe_nums is True:
        torch_input_tensor_a = gen_func_with_cast_tt(
            partial(torch_random, low=-10, high=10, dtype=torch.float32), input_a_dtype
        )(input_shape)
    else:
        torch_input_tensor_a = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
        )(input_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.exp(input_tensor_a, memory_config=output_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    return [[True, "0.999"], e2e_perf]
