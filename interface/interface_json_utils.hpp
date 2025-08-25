#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

namespace op_perf {

nlohmann::json load_op_category_map(const std::string &json_path);

} // namespace op_perf
