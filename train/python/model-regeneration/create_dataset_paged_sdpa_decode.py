# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import ast
import create_dataset_utils as utils

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

#the dataset from sweeps packs valid combinations of shapes and some other parameters into one, example:
# "valid_combination": "((1, 8, 40, 128), (8, 8, 64, 128), (8, 8, 64, 128), (8, 2), (8, 1, 40, 128), (8,), False, 32, 11.313708498984761)"
# q_shape, k_shape, v_shape, page_table_shape, attn_mask_shape, cur_pos_shape, is_causal, k_chunk, scale respectively
#this function will take in the valid_combinations json object and unpack this into individual parameters
def unpack_datapoint(datapoint):

    valid_combination = datapoint["valid_combination"]

    # Unpack the valid_combination string into a tuple using ast.literal_eval for safety
    unpacked = ast.literal_eval(valid_combination)

    # Map the unpacked values to their respective variables
    (q_shape, k_shape, v_shape, page_table_shape, attn_mask_shape, cur_pos_shape, is_causal, k_chunk, scale) = unpacked

    return {
        "q_shape": q_shape,
        "k_shape": k_shape,
        "v_shape": v_shape,
        "page_table_shape": page_table_shape,
        "attn_mask_shape": attn_mask_shape,
        "cur_pos_shape": cur_pos_shape,
        "is_causal": is_causal,
        "k_chunk": k_chunk,
        "scale": scale
    }

#Get tensor shape, dtype, and memory config (L1 / DRAM) from sweep vectors.
#Returns what?
def get_shapes_dtype_memcfgs(test_vectors):
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

    input_parameters = get_shapes_dtype_memcfgs(test_vectors) #to complete

    kernel_duration = utils.get_kernel_durations(results)

    create_dataset_csv(op_name, input_parameters, kernel_duration)