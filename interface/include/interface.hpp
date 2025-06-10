#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>

#include <nlohmann/json.hpp>

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

using namespace mlpack;

FFN<MeanSquaredError, RandomInitialization> load_mlpack_model(
    const std::string& model_path,
    int input_size,
    const std::vector<int>& hidden_layers
);

template<typename ... Args>
uint64_t get_runtime_from_model(
    const std::string& op_name,
    Args&& ... args
);
