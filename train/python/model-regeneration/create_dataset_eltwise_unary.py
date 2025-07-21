import create_dataset_utils as utils

#Get tensor shape, dtype, and memory config (L1 / DRAM) from sweep vectors
def get_shape_dtype_memcfg(test_vectors):

    input_shape = [[], [], [], []] #max tensor rank is 4
    dtype = [[], [], [], [], []] #bfloat8,bfloat16,float32,uint16,uint32
    input_mem_config = [[], []] #DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG

    for suite in test_vectors.keys():
        vectors = test_vectors[suite]
        for id in vectors.keys():
            dict = vectors[id]
            input = utils.parse_ints_from_string(dict["input_shape"])
            input_dtype = dict["input_a_dtype"]
            input_memcfg = dict["input_a_memory_config"]
            input_memcfg = input_memcfg["data"]
            #get input shape
            for i in range(4): # is max num of dimensions of tensor
                if i < len(input):
                    input_shape[i].append(input[i])
                else:
                    input_shape[i].append(0)
            #get dtype in one-hot encoding
            if input_dtype == "DataType.BFLOAT8_B":
                dtype[0].append(1)
                dtype[1].append(0)
                dtype[2].append(0)
                dtype[3].append(0)
                dtype[4].append(0)
            elif input_dtype == "DataType.BFLOAT16":
                dtype[0].append(0)
                dtype[1].append(1)
                dtype[2].append(0)
                dtype[3].append(0)
                dtype[4].append(0)
            elif input_dtype == "DataType.FLOAT32":
                dtype[0].append(0)
                dtype[1].append(0)
                dtype[2].append(1)
                dtype[3].append(0)
                dtype[4].append(0)
            elif input_dtype == "DataType.UINT16":
                dtype[0].append(0)
                dtype[1].append(0)
                dtype[2].append(0)
                dtype[3].append(1)
                dtype[4].append(0)
            elif input_dtype == "DataType.UINT32":
                dtype[0].append(0)
                dtype[1].append(0)
                dtype[2].append(0)
                dtype[3].append(0)
                dtype[4].append(1)
            #get memcfg in one hot encodings
            if input_memcfg == "{\"buffer_type\":0,\"memory_layout\":0}":
                input_mem_config[0].append(1)
                input_mem_config[1].append(0)
            if input_memcfg == "{\"buffer_type\":1,\"memory_layout\":0}":
                input_mem_config[0].append(0)
                input_mem_config[1].append(1)
    
    return input_shape, dtype, input_mem_config

def create_dataset_csv(op_name, input_shape, dtype, input_mem_config, kernel_duration):

    with open(f"{op_name}_dataset.csv", 'w') as f:
        for i in range(len(input_shape[0])):
            if kernel_duration[i] == -1:
                continue
            f.write(f"{input_shape[0][i]},{input_shape[1][i]},{input_shape[2][i]},{input_shape[3][i]},{dtype[0][i]},{dtype[1][i]},{dtype[2][i]},{dtype[3][i]},{dtype[4][i]},{input_mem_config[0][i]},{input_mem_config[1][i]},{kernel_duration[i]}\n")

def create_dataset_eltwise_unary(op_name, sweep_test_vectors, sweep_results):
    
    results, test_vectors = utils.load_json(sweep_test_vectors, sweep_results)

    input_shape, dtype, input_mem_config = get_shape_dtype_memcfg(test_vectors)

    kernel_duration = utils.get_kernel_durations(results)

    create_dataset_csv(op_name, input_shape, dtype,input_mem_config, kernel_duration)