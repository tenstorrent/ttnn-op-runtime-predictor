// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace op_perf {

nlohmann::json load_op_category_map(const std::string &json_path);

} // namespace op_perf
