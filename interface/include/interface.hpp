#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>

#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include <nlohmann/json.hpp>

using namespace mlpack;

FFN<MeanSquaredError, RandomInitialization> load_mlpack_model(
    const std::string& model_path,
    int input_size,
    const std::vector<int>& hidden_layers
);

template<typename ... Args>
uint64_t get_runtime_from_model(const std::string& op_name, Args&& ... args){

    //determine op
    if (op_name == "ttnn::exp") {
        //return predict_exp_runtime(std::forward<Args>(args)...);
        return 1;
    } else {
        return 0;
    }
}
