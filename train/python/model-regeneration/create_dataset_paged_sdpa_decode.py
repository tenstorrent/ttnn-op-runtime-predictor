# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import ast
import create_dataset_utils as utils

PAGED_SDPA_DECODE_NUM_SUPPORTED_DTYPES = 2 #bfloat8, bfloat16

def append_input_shape(input_shape_list, input_shape):
    assert len(input_shape) == utils.MAX_TENSOR_RANK, "tensor rank must be 4"
    input_shape_list.append(input_shape)

# concat heads supports only bfloat8 and bfloat16
def append_dtype(dtype_list, input_dtype):
    dtype = [0] * PAGED_SDPA_DECODE_NUM_SUPPORTED_DTYPES
    if input_dtype == "DataType.BFLOAT8_B":
        dtype[0] = 1
    elif input_dtype == "DataType.BFLOAT16":
        dtype[1] = 1
    else:
        raise ValueError("Error: datatype is unspecified")
    dtype_list.append(dtype)

def append_memory_config(input_memcfg_list, input_memcfg):
    memcfg = [0] * (utils.NUM_BUFFER_TYPES)

    if input_memcfg == "{\"buffer_type\":0,\"created_with_nd_shard_spec\":false,\"memory_layout\":0}":
        memcfg[0] = 1 #DRAM
    elif input_memcfg == "{\"buffer_type\":1,\"created_with_nd_shard_spec\":false,\"memory_layout\":2}":
        memcfg[1] = 1 #L1_HEIGHT_SHARDED
    else:
        raise ValueError("Error: memory config buffer type is unspecified")
    input_memcfg_list.append(memcfg) #only DRAM, L1_HEIGHT_SHARDED for paged_sdpa_decode

def append_math_fidelity(math_fidelity_list, math_fidelity):
    fidelity = []
    if math_fidelity == "MathFidelity.LoFi":
        fidelity = 1
    elif math_fidelity == "MathFidelity.HiFi2":
        fidelity = 2
    elif math_fidelity == "MathFidelity.HiFi3":
        fidelity = 3
    elif math_fidelity == "MathFidelity.HiFi4":
        fidelity = 4
    else:
        raise ValueError("Error: math fidelity is unspecified")
    math_fidelity_list.append(fidelity)

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

    if attn_mask_shape == None:
        attn_mask_shape = (0, 0, 0, 0)
    if cur_pos_shape == None:
        cur_pos_shape = (0,)

    if is_causal == True:
        is_causal = 1
    else:
        is_causal = 0

    return {
        "q_shape": q_shape,
        "k_shape": k_shape,
        "v_shape": v_shape,
        "page_table_shape": page_table_shape,
        "attn_mask_shape": attn_mask_shape,
        "cur_pos_shape": cur_pos_shape,
        "is_causal": is_causal,
        "k_chunk_size": k_chunk,
        "scale": scale
    }

#Get tensor shape, dtype, and memory config (L1 / DRAM) from sweep vectors.
#Returns what?
def get_shapes_dtype_memcfgs(test_vectors):

    datapoint_list = [] #see unpack_datapoint() above
    dtype_list = [] #bfloat8,bfloat16
    input_memcfg_list = [] #DRAM_MEMORY_CONFIG, L1_HEIGHT_SHARDED_MEMORY_CONFIG
    output_memcfg_list = [] #DRAM_MEMORY_CONFIG, L1_HEIGHT_SHARDED_MEMORY_CONFIG
    math_fidelity_list = [] #MathFidelity.HiFi2
    math_approx_mode_list = [] #True, False
    fp32_dest_acc_en_list = [] #True, False
    packer_l1_acc_list = [] #True, False
    exp_approx_mode_list = [] #True, False
    q_chunk_size_list = [] #[32, 64, 128, 256, 512, 1024, 2048, 4096]
    use_sdpa_program_config_list = [] #True, False
    use_device_compute_kernel_config_list = [] #True, False

    for suite in test_vectors.keys():
        vectors = test_vectors[suite]
        for id in vectors.keys():

            vector_dict = vectors[id]

            datapoint = unpack_datapoint(vector_dict)

            input_dtype = vector_dict["input_dtype"]
            input_q_memcfg = vector_dict["input_q_memory_config"]["data"]
            output_memcfg = vector_dict["output_memory_config"]["data"]

            math_fidelity = vector_dict["math_fidelity"]
            math_approx_mode = vector_dict["math_approx_mode"]
            fp32_dest_acc_en = vector_dict["fp32_dest_acc_en"]
            packer_l1_acc = vector_dict["packer_l1_acc"]
            exp_approx_mode = vector_dict["exp_approx_mode"]
            q_chunk_size = int(vector_dict["q_chunk_size"])
            use_sdpa_program_config = vector_dict["use_sdpa_program_config"]
            use_device_compute_kernel_config = vector_dict["use_device_compute_kernel_config"]

            if use_sdpa_program_config == "False":
                datapoint["k_chunk_size"] = -1 
                q_chunk_size_list.append(-1)
                exp_approx_mode_list.append(-1)
            else:
                q_chunk_size_list.append(int(q_chunk_size))
                exp_approx_mode_list.append(1 if exp_approx_mode == "True" else 0)

            if use_device_compute_kernel_config == "False":
                math_fidelity_list.append(-1)
                math_approx_mode_list.append(-1)
                fp32_dest_acc_en_list.append(-1)
                packer_l1_acc_list.append(-1)
            else:
                append_math_fidelity(math_fidelity_list, math_fidelity)
                math_approx_mode_list.append(1 if math_approx_mode == "True" else 0)
                fp32_dest_acc_en_list.append(1 if fp32_dest_acc_en == "True" else 0)
                packer_l1_acc_list.append(1 if packer_l1_acc == "True" else 0)

            datapoint_list.append(datapoint)
            append_dtype(dtype_list, input_dtype)
            append_memory_config(input_memcfg_list, input_q_memcfg)
            append_memory_config(output_memcfg_list, output_memcfg)

            use_sdpa_program_config_list.append(1 if use_sdpa_program_config == "True" else 0)
            use_device_compute_kernel_config_list.append(1 if use_device_compute_kernel_config == "True" else 0)

    return {
        "datapoints": datapoint_list,
        "dtypes": dtype_list,
        "input_memcfgs": input_memcfg_list,
        "output_memcfgs": output_memcfg_list,
        "math_fidelity": math_fidelity_list,
        "math_approx_mode": math_approx_mode_list,
        "fp32_dest_acc_en": fp32_dest_acc_en_list,
        "packer_l1_acc": packer_l1_acc_list,
        "exp_approx_mode": exp_approx_mode_list,
        "q_chunk_size": q_chunk_size_list,
        "use_sdpa_program_config": use_sdpa_program_config_list,
        "use_device_compute_kernel_config": use_device_compute_kernel_config_list
    }

def create_dataset_csv(op_name, input_parameters, kernel_duration_list):

    with open(f"{op_name}_dataset.csv", 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow([
            'q_shape_1','q_shape_2','q_shape_3','q_shape_4',
            'k_shape_1','k_shape_2','k_shape_3','k_shape_4',
            'v_shape_1','v_shape_2','v_shape_3','v_shape_4',
            'page_table_shape_1','page_table_shape_2',
            'attn_mask_shape_1','attn_mask_shape_2','attn_mask_shape_3','attn_mask_shape_4',
            'cur_pos_shape_1',
            'bfloat8','bfloat16',
            'input_DRAM','input_L1_height_sharded',
            'output_DRAM','output_L1_height_sharded',
            'is_causal',
            'scale',
            'q_chunk_size',
            'k_chunk_size',
            'exp_approx_mode',
            'use_sdpa_program_config',
            'math_fidelity',
            'math_approx_mode',
            'fp32_dest_acc_en',
            'packer_l1_acc',
            'use_device_compute_kernel_config',
            'kernel_duration'
        ])

        print(f"Number of datapoints to write: {len(input_parameters['datapoints'])}")
        print(f"Number of kernel durations to write: {len(kernel_duration_list)}")
        for i in range(len(kernel_duration_list)):
            if kernel_duration_list[i] == -1:
                continue

            csv_writer.writerow(
                list(input_parameters["datapoints"][i]["q_shape"]) + 
                list(input_parameters["datapoints"][i]["k_shape"]) +
                list(input_parameters["datapoints"][i]["v_shape"]) +
                list(input_parameters["datapoints"][i]["page_table_shape"]) +
                list(input_parameters["datapoints"][i]["attn_mask_shape"]) +
                list(input_parameters["datapoints"][i]["cur_pos_shape"]) +
                input_parameters["dtypes"][i] +
                input_parameters["input_memcfgs"][i] +
                input_parameters["output_memcfgs"][i] +
                [input_parameters["datapoints"][i]["is_causal"]] + 
                [input_parameters["datapoints"][i]["scale"]] +
                [input_parameters["q_chunk_size"][i]] +
                [input_parameters["datapoints"][i]["k_chunk_size"]] +
                [input_parameters["exp_approx_mode"][i]] +
                [input_parameters["use_sdpa_program_config"][i]] +
                [input_parameters["math_fidelity"][i]] +
                [input_parameters["math_approx_mode"][i]] +
                [input_parameters["fp32_dest_acc_en"][i]] +
                [input_parameters["packer_l1_acc"][i]] +
                [input_parameters["use_device_compute_kernel_config"][i]] +
                [kernel_duration_list[i]]
            )

def create_dataset_paged_sdpa_decode(op_name, sweep_test_vectors, sweep_results):
    
    results, test_vectors = utils.load_json(sweep_test_vectors, sweep_results)

    input_parameters = get_shapes_dtype_memcfgs(test_vectors) #to complete

    kernel_duration = utils.get_kernel_durations(results)

    create_dataset_csv(op_name, input_parameters, kernel_duration)

""" "fac7e831d396127ec970dfc0cc6eadeb5f06efd42049fa860f83ac7d": {
      "valid_combination": "((1, 8, 40, 128), (8, 8, 64, 128), (8, 8, 64, 128), (8, 2), (8, 1, 40, 128), (8,), False, 32, 11.313708498984761)",
      "input_dtype": "DataType.BFLOAT16",
      "input_q_memory_config": {
        "type": "ttnn._ttnn.tensor.MemoryConfig",
        "data": "{\"buffer_type\":1,\"created_with_nd_shard_spec\":false,\"memory_layout\":2}"
      },
      "output_memory_config": {
        "type": "ttnn._ttnn.tensor.MemoryConfig",
        "data": "{\"buffer_type\":1,\"created_with_nd_shard_spec\":false,\"memory_layout\":2}"
      },
      "math_fidelity": "MathFidelity.HiFi2",
      "math_approx_mode": "True",
      "fp32_dest_acc_en": "True",
      "packer_l1_acc": "True",
      "exp_approx_mode": "True",
      "q_chunk_size": "32",
      "use_sdpa_program_config": "True",
      "use_device_compute_kernel_config": "True",
      "suite_name": "suite_1",
      "validity": "VectorValidity.INVALID",
      "invalid_reason": "Invalid: L1_HEIGHT_SHARDED_MEMORY_CONFIG requires batch size = 1",
      "status": "VectorStatus.CURRENT",
      "sweep_name": "simple_paged_sdpa_decode_sweep",
      "timestamp": "2025-10-14_01-12-40",
      "input_hash": "fac7e831d396127ec970dfc0cc6eadeb5f06efd42049fa860f83ac7d",
      "tag": "brapanan"
    }, """