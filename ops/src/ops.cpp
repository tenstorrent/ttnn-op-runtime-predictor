// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/include/ops.hpp"
#include "ops/include/model.hpp"

namespace op_perf {

std::optional<
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>>
load_mlpack_model(const std::string &model_path, int input_size,
                  const std::vector<int> &hidden_layers) {

  // initialize model architecture
  mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> model;
  model.InputDimensions() =
      std::vector<size_t>({static_cast<unsigned long>(input_size)});

  const int num_hidden_layers = hidden_layers.size();
  for (int i = 0; i < num_hidden_layers; ++i) {
    model.Add<mlpack::Linear>(hidden_layers[i]);
    model.Add<mlpack::ReLU>();
  }
  model.Add<mlpack::Linear>(1);

  // load model from path
  try {
    mlpack::data::Load(model_path, "model", model, true);
  } catch (const std::exception &e) {
    return std::nullopt;
  }

  return model;
}

std::vector<int> get_tensor_dimensions(const nlohmann::json &tensor_dim_array) {
  std::vector<int> dim_vector = tensor_dim_array;
  const int length = dim_vector.size();
  constexpr int max_rank = 4;

  for (int i = length; i < max_rank; ++i) {
    dim_vector.push_back(0);
  }

  return dim_vector;
}

std::vector<int> get_one_hot_dtype(const int &dtype) {

  // vector datatypes in order: BFLOAT8_B, BFLOAT16, FLOAT32, UINT16, UINT32
  std::vector<int> onehot_dtype(5, 0);

  switch (dtype) {
  case BFLOAT8_B:
    onehot_dtype[0] = 1;
    break;
  case BFLOAT16:
    onehot_dtype[1] = 1;
    break;
  case FLOAT32:
    onehot_dtype[2] = 1;
    break;
  case UINT16:
    onehot_dtype[3] = 1;
    break;
  case UINT32:
    onehot_dtype[4] = 1;
    break;
  default:
    break;
  }

  return onehot_dtype;
}

std::vector<int> get_memory_config(const int &memory_config) {

  // memory_config in order: DRAM, L1
  std::vector<int> mem_cfg_vector(2, 0);

  if (memory_config == DRAM) {
    mem_cfg_vector[0] = 1;
  } else {
    // L1
    mem_cfg_vector[1] = 1;
  }

  return mem_cfg_vector;
}

uint64_t predict_exp_runtime(const nlohmann::json &tensor_json,
                             const nlohmann::json &optional_output_layout) {

  if (tensor_json.is_null() || tensor_json.is_number()) {
    return 0;
  }

  // set exp model parameters and model path
  const int input_size = 11;
  const std::vector<int> hidden_layers = {128, 128, 128};
  const std::string model_path = std::string(MODEL_PATH) + "exp_mlp_model.bin";
  const std::string scaler_path = std::string(MODEL_PATH) + "exp_scaler.bin";

  // load mlp
  auto model_optional =
      load_mlpack_model(model_path, input_size, hidden_layers);
  if (!model_optional.has_value()) {
    return 0;
  }
  auto &model = *model_optional;

  mlpack::data::StandardScaler scaler;
  try {
    mlpack::data::Load(scaler_path, "scaler", scaler, true);
  } catch (std::exception &e) {
    return 0;
  }

  // get input, process it into arma::vec

  // specify dimension
  if (tensor_json["logical_shape"].size() < 2 ||
      tensor_json["logical_shape"].size() > 4) {
    // max allowed tensor dim is 4
    return 0;
  }
  std::vector<int> tensor_dim_array =
      get_tensor_dimensions(tensor_json["logical_shape"]);

  // specify datatype
  int ttnn_tensor_dtype =
      tensor_json["tensor_layout"]["dtype"];
  std::vector<int> onehot_dtype = get_one_hot_dtype(ttnn_tensor_dtype);

  // specify memory_config
  int mem_cfg = tensor_json["tensor_layout"]["memory_config"]["buffer_type"];
  std::vector<int> memory_config = get_memory_config(mem_cfg);

  // create input and output vectors
  arma::vec input = {

      static_cast<double>(tensor_dim_array[0]),
      static_cast<double>(tensor_dim_array[1]),
      static_cast<double>(tensor_dim_array[2]),
      static_cast<double>(tensor_dim_array[3]),
      static_cast<double>(onehot_dtype[0]),
      static_cast<double>(onehot_dtype[1]),
      static_cast<double>(onehot_dtype[2]),
      static_cast<double>(onehot_dtype[3]),
      static_cast<double>(onehot_dtype[4]),
      static_cast<double>(memory_config[0]),
      static_cast<double>(memory_config[1])

  };

  arma::vec scaler_scaled;
  scaler.Transform(input, scaler_scaled);

  // model inference
  arma::mat scaler_output;
  model.Predict(scaler_scaled, scaler_output);

  // some small tensors have small runtime, and the runtime inference may be
  // negative. In this case, return 0.
  if (scaler_output(0, 0) < 0) {
    return 0;
  }
  return static_cast<uint64_t>(scaler_output(0, 0));
}

} // namespace op_perf