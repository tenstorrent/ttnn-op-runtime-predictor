// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops.hpp"
#include "interface_json_utils.hpp"
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace op_perf {

template <typename... Args>
uint64_t get_runtime_from_model(const std::string &op_name, Args &&...args) {

  try {
    const nlohmann::json op_category_json = op_perf::load_op_category_map(
        std::string(OP_CATEGORY_PATH) + "op_categories.json");
    if (!op_category_json.contains(op_name)) {
      throw std::runtime_error("Error: Unknown op name: " + op_name);
    }
    const std::string &category = op_category_json[op_name];

    if (category == "eltwise_unary") {
      return predict_eltwise_unary_runtime(op_name,
                                           std::forward<Args>(args)...);
    } else {
      return 0;
    }
  } catch (const std::exception &) {
    return 0;
  }
}

} // namespace op_perf
