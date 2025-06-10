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

#include <armadillo>

uint64_t predict_exp_runtime(
    const nlohmann::json& tensor_and_shape_jsons,
    const nlohmann::json& optional_output_layout
);