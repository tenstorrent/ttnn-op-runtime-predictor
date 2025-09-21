# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import create_dataset_utils as utils

CREATE_QKV_HEADS_NUM_SUPPORTED_DTYPES = 2

def append_input_shape(input_shape_list, input_shape):
    assert len(input_shape) == utils.MAX_TENSOR_RANK, "tensor rank must be 4"
    input_shape_list.append(input_shape)

# create qkv heads supports only bfloat8 and bfloat16
def append_dtype(dtype_list, input_dtype):
    dtype = [0] * CREATE_QKV_HEADS_NUM_SUPPORTED_DTYPES
    if input_dtype == "DataType.BFLOAT8_B":
        dtype[0] = 1
    elif input_dtype == "DataType.BFLOAT16":
        dtype[1] = 1
    else:
        raise ValueError("Error: datatype is unspecified")
    dtype_list.append(dtype)

#Get tensor shape, dtype, and memory config (L1 / DRAM) from sweep vectors.
#Returns 3 lists, containing input shapes, ex. [1, 1, 32, 32], datatypes in one hot encoding, ex. [0, 1, 0, 0, 0], and memory config buffer type in one hot encoding, ex. [0,1]
def get_shape_dtype_memcfg(test_vectors):
    assert isinstance(test_vectors, dict), "The object is not a dictionary."

    input_shape_list = [] #tensor rank must be 4
    dtype_list = [] #bfloat8,bfloat16

    for suite in test_vectors.keys():
        vectors = test_vectors[suite]
        for id in vectors.keys():

            vector_dict = vectors[id]

            input = utils.parse_ints_from_string(vector_dict["input_shape"])
            input_dtype = vector_dict["input_dtype"]

            append_input_shape(input_shape_list, input)
            append_dtype(dtype_list, input_dtype)

    return input_shape_list, dtype_list

def create_dataset_csv(op_name, input_shape_list, dtype_list, kernel_duration_list):

    with open(f"{op_name}_dataset.csv", 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['input_shape_1','input_shape_2','input_shape_3','input_shape_4','bfloat8','bfloat16','kernel_duration'])

        for i in range(len(input_shape_list)):
            if kernel_duration_list[i] == -1:
                continue
            csv_writer.writerow(input_shape_list[i] + dtype_list[i] + [kernel_duration_list[i]])

def create_dataset_create_qkv_heads(op_name, sweep_test_vectors, sweep_results):
    
    results, test_vectors = utils.load_json(sweep_test_vectors, sweep_results)

    input_shape, dtype = get_shape_dtype_memcfg(test_vectors)

    kernel_duration = utils.get_kernel_durations(results)

    create_dataset_csv(op_name, input_shape, dtype, kernel_duration)
