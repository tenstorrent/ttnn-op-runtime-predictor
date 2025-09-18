# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import create_dataset_utils as utils

CONCAT_HEADS_NUM_SUPPORTED_DTYPES = 2

def append_input_shape(input_shape_list, input_shape):
    assert len(input_shape) == utils.MAX_TENSOR_RANK, "tensor rank must be 4"
    input_shape_list.append(input_shape)

# concat heads supports only bfloat8 and bfloat16
def append_dtype(dtype_list, input_dtype):
    dtype = [0] * CONCAT_HEADS_NUM_SUPPORTED_DTYPES
    if input_dtype == "DataType.BFLOAT8_B":
        dtype[0] = 1
    elif input_dtype == "DataType.BFLOAT16":
        dtype[1] = 1
    else:
        raise ValueError("Error: datatype is unspecified")
    dtype_list.append(dtype)

def append_memory_config(input_memcfg_list, input_memcfg):
    memcfg = [0] * (utils.NUM_BUFFER_TYPES + 1)

    if input_memcfg == "{\"buffer_type\":0,\"created_with_nd_shard_spec\":false,\"memory_layout\":0}":
        memcfg[0] = 1 #DRAM
    elif input_memcfg == "{\"buffer_type\":1,\"created_with_nd_shard_spec\":false,\"memory_layout\":0}":
        memcfg[1] = 1 #L1
    elif input_memcfg == "{\"buffer_type\":1,\"created_with_nd_shard_spec\":false,\"memory_layout\":2}":
        memcfg[2] = 1 #L1_HEIGHT_SHARDED
    else:
        raise ValueError("Error: memory config buffer type is unspecified")
    input_memcfg_list.append(memcfg[0:2]) #only L1, DRAM

#Get tensor shape, dtype, and memory config (L1 / DRAM) from sweep vectors.
#Returns 3 lists, containing input shapes, ex. [1, 1, 32, 32], datatypes in one hot encoding, ex. [0, 1, 0, 0, 0], and memory config buffer type in one hot encoding, ex. [0,1]
def get_shape_dtype_memcfg(test_vectors):
    assert isinstance(test_vectors, dict), "The object is not a dictionary."

    input_shape_list = [] #tensor rank must be 4
    dtype_list = [] #bfloat8,bfloat16
    input_mem_cfg_list = [] #DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG, L1_HEIGHT_SHARDED_MEMORY_CONFIG
    output_memcfg_list = [] #DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG, L1_HEIGHT_SHARDED_MEMORY_CONFIG

    for suite in test_vectors.keys():
        vectors = test_vectors[suite]
        for id in vectors.keys():

            vector_dict = vectors[id]

            input = utils.parse_ints_from_string(vector_dict["input_shape"])
            input_dtype = vector_dict["input_dtype"]
            input_memcfg = vector_dict["input_memory_config"]
            input_memcfg = input_memcfg["data"]
            output_memcfg = vector_dict["output_memory_config"]
            output_memcfg = output_memcfg["data"]

            append_input_shape(input_shape_list, input)
            append_dtype(dtype_list, input_dtype)
            append_memory_config(input_mem_cfg_list, input_memcfg)
            append_memory_config(output_memcfg_list, output_memcfg)

    return input_shape_list, dtype_list, input_mem_cfg_list, output_memcfg_list

def create_dataset_csv(op_name, input_shape_list, dtype_list, input_mem_cfg_list, output_memcfg_list, kernel_duration_list):

    with open(f"{op_name}_dataset.csv", 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['input_shape_1','input_shape_2','input_shape_3','input_shape_4','bfloat8','bfloat16','input_DRAM','input_L1','output_DRAM','output_L1','kernel_duration'])

        for i in range(len(input_shape_list)):
            if kernel_duration_list[i] == -1:
                continue
            csv_writer.writerow(input_shape_list[i] + dtype_list[i] + input_mem_cfg_list[i] + output_memcfg_list[i] + [kernel_duration_list[i]])

def create_dataset_concatenate_heads(op_name, sweep_test_vectors, sweep_results):
    
    results, test_vectors = utils.load_json(sweep_test_vectors, sweep_results)

    input_shape, dtype, input_mem_config, output_mem_config = get_shape_dtype_memcfg(test_vectors)

    kernel_duration = utils.get_kernel_durations(results)

    create_dataset_csv(op_name, input_shape, dtype, input_mem_config, output_mem_config, kernel_duration)
