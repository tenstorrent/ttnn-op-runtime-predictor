// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace op_perf {

typedef enum Dtype {

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

typedef enum mem_cfg {

  DRAM = 0,
  L1 = 1,

} mem_cfg;

std::vector<int> get_tensor_dimensions(const nlohmann::json& tensor_dim_array);
std::vector<int> get_one_hot_dtype(const int &dtype);
std::vector<int> get_memory_config(const int &memory_config);

/* uint64_t predict_exp_runtime(const nlohmann::json &tensor_and_shape_jsons,
                             const nlohmann::json &optional_output_layout); */

uint64_t predict_eltwise_unary_runtime(const std::string& op_name, 
    const nlohmann::json &tensor_and_shape_jsons,
    const nlohmann::json &optional_output_layout);

} // namespace op_perf