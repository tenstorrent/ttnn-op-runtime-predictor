#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ops/include/ops.hpp"

namespace op_perf {

template <typename... Args>
uint64_t get_runtime_from_model(const std::string &op_name, Args &&...args) {

  if (op_name == "ttnn::exp") {
    // return op_perf::predict_exp_runtime(std::forward<Args>(args)...);
    return 1; // for test purposes
  } else {
    return 0;
  }
}

} //namespace op_perf