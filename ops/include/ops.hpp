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

typedef enum {
    
    BFLOAT16 = 0,
    FLOAT32 = 1,
    UINT32 = 2,
    BFLOAT8_B = 3,
    BFLOAT4_B = 4,
    UINT8 = 5,
    UINT16 = 6,
    INT32 = 7,
    INVALID = 8,

} DType;

uint64_t predict_exp_runtime(
    const nlohmann::json& tensor_and_shape_jsons,
    const nlohmann::json& optional_output_layout
);