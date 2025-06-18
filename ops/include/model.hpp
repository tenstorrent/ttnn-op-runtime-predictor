#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include <armadillo>

namespace op_perf{

std::optional<
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>>
load_mlpack_model(const std::string &model_path, const int input_size,
                  const std::vector<int> &hidden_layers);

}//namespace op_perf