#include "../include/ops.hpp"

uint64_t predict_exp_runtime(nlohmann::json tensor_and_shape_jsons, nlohmann::json optional_output_layout){
    //set exp model parameters and model path
    int input_size = 11;
    std::vector<int> hidden_layers = {128, 128, 128};
    std::string model_path = "mlp_folder/exp_model.bin"; //placeholder filepath

    //load mlp
    auto model_optional = load_mlpack_model(model_path, input_size, hidden_layers);
    if(!model_optional.has_value()){
        return 0;
    }
    auto& model = *model_optional;

    //get input, process it into arma::vec. This is likely to change when exp mlp is trained
    //specify dimension
    auto dim_list = tensor_and_shape_jsons[1];
    int dim1 = 0, dim2 = 0, dim3 = 0, dim4 = 0;
    int size = dim_list.size();
    if(size >= 1) dim1 = dim_list[0];
    if(size >= 2) dim2 = dim_list[1];
    if(size >= 3) dim3 = dim_list[2];
    if(size >= 4) dim4 = dim_list[3];

    //specify datatype
    int ttnn_tensor_dtype = tensor_and_shape_jsons[0]["tensor_spec"]["tensor_layout"]["dtype"];
    int bfloat16 = 0, float32 = 0, uint32 = 0, bfloat8_b = 0, uint16 = 0;
    switch(ttnn_tensor_dtype){
        case 0: 
            bfloat16 = 1; 
            break;
        case 1: 
            float32 = 1; 
            break;
        case 2: 
            uint32 = 1; 
            break;
        case 3:    
            bfloat8_b = 1; 
            break;
        case 6: 
            uint16 = 1; 
            break;
        default: 
            break;
    }

    //specify dram or l1 memory_config
    int dram = 0, l1 = 0;
    auto mem_cfg = tensor_and_shape_jsons[0]["tensor_spec"]["tensor_layout"]["memory_config"]["buffer_type"];
    if(mem_cfg == 0){
        dram = 1;
    }else{
        l1 = 1;
    }

    //create input and output vectors. This is subject to change based on how categorical data is encoded in model
    //current implementation is onehot encoding for datatype and memory_config
    arma::vec input = {

        static_cast<double>(dim1),
        static_cast<double>(dim2),
        static_cast<double>(dim3),
        static_cast<double>(dim4),
        static_cast<double>(bfloat8_b),
        static_cast<double>(bfloat16),
        static_cast<double>(float32),
        static_cast<double>(uint16),
        static_cast<double>(uint32),
        static_cast<double>(dram),
        static_cast<double>(l1)
    };
    arma::vec output(1);

    //model inference
    model.Predict(input, output);
    
    return output(0);

}