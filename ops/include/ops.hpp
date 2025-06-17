#pragma once

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <armadillo>
#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <nlohmann/json.hpp>

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

std::optional<
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>>
load_mlpack_model(const std::string &model_path, const int input_size,
                  const std::vector<int> &hidden_layers);

std::vector<int> get_one_hot_dtype(const int &dtype);
std::vector<int> get_memory_config(const int &memory_config);

uint64_t predict_exp_runtime(const nlohmann::json &tensor_and_shape_jsons,
                             const nlohmann::json &optional_output_layout);