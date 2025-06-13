#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <optional>

#include <nlohmann/json.hpp>

#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include <armadillo>

#include "../../interface/include/interface.hpp"

uint64_t predict_exp_runtime(
    const nlohmann::json& tensor_and_shape_jsons,
    const nlohmann::json& optional_output_layout
);