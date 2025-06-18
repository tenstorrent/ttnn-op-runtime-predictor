#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <iostream>

#include <nlohmann/json.hpp>

namespace op_perf{

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

std::vector<int> get_one_hot_dtype(const int &dtype);
std::vector<int> get_memory_config(const int &memory_config);

uint64_t predict_exp_runtime(const nlohmann::json &tensor_and_shape_jsons,
                             const nlohmann::json &optional_output_layout);

}//namespace op_perf