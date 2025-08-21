// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../ops/include/ops.hpp"

namespace op_perf {

template <typename... Args>
uint64_t get_runtime_from_model(const std::string &op_name, Args &&...args) {

  if (op_name == "exp") {
    return predict_eltwise_unary_runtime(op_name, std::forward<Args>(args)...);
  } else {
    return 0;
  }
}

} // namespace op_perf
