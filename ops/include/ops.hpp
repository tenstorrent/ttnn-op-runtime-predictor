// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

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
  L1_HEIGHT_SHARDED = 2,

} mem_cfg;

std::vector<int> get_tensor_dimensions(const nlohmann::json& tensor_dim_array);
std::vector<int> get_one_hot_dtype(const int &dtype);
std::vector<int> get_memory_config(const int &memory_config);

uint64_t predict_eltwise_unary_runtime(const std::string& op_name, 
    const nlohmann::json &tensor_json,
    const nlohmann::json &optional_output_layout);

uint64_t predict_concatenate_heads_runtime(const nlohmann::json &tensor_json,
    const nlohmann::json &optional_output_layout);

uint64_t predict_create_qkv_heads_runtime(const nlohmann::json &tensor_json,
    const int &num_heads,
    const std::optional<int>& num_kv_heads,
    const bool &transpose_k_heads
    );

uint64_t predict_paged_sdpa_decode_runtime(
  const nlohmann::json &q_tensor_json,
  const nlohmann::json &k_tensor_json,
  const nlohmann::json &v_tensor_json,
  const nlohmann::json &page_table_tensor_json,
  const bool &is_causal,
  const std::optional<nlohmann::json> &optional_attn_mask_tensor_json,
  const bool& cur_pos_empty,
  const std::optional<nlohmann::json> &optional_cur_pos_tensor_json,
  const float &optional_scale,
  const nlohmann::json &optional_output_memory_config,
  const nlohmann::json &optional_compute_kernel_config
  );

} // namespace op_perf