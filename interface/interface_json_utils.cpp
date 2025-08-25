#include "interface_json_utils.hpp"

namespace op_perf {

nlohmann::json load_op_category_map(const std::string &json_path) {
  std::ifstream infile(json_path);
  if (!infile) {
    throw std::runtime_error("Error: could not open file: " + json_path);
  }
  nlohmann::json op_map;
  infile >> op_map;
  return op_map;
}

} // namespace op_perf
