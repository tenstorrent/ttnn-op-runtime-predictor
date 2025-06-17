#include "ops/include/ops.hpp"
#include "ops/include/model.hpp"

namespace op_perf{

std::optional<
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>>
load_mlpack_model(const std::string &model_path, int input_size,
                  const std::vector<int> &hidden_layers) {

  // initialize model architecture
  mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> model;
  model.InputDimensions() =
      std::vector<size_t>({static_cast<unsigned long>(input_size)});

  int num_hidden_layers = hidden_layers.size();
  for (int i = 0; i < num_hidden_layers; i++) {
    model.Add<mlpack::Linear>(hidden_layers[i]);
    model.Add<mlpack::ReLU>();
  }
  model.Add<mlpack::Linear>(1);

  // load model from path
  try {
    mlpack::data::Load(model_path, "model", model);
  } catch (const std::exception &e) {
    return std::nullopt;
  }

  return model;
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

  if (memory_config == 0) {
    mem_cfg_vector[0] = 1;
  } else {
    mem_cfg_vector[1] = 1;
  }

  return mem_cfg_vector;
}

uint64_t predict_exp_runtime(const nlohmann::json &tensor_and_shape_jsons,
                             const nlohmann::json &optional_output_layout) {
  // set exp model parameters and model path
  const int input_size = 11;
  const std::vector<int> hidden_layers = {128, 128, 128};
  const std::string model_path =
      "mlp_folder/exp_model.bin"; // placeholder filepath

  // load mlp
  auto model_optional =
      op_perf::load_mlpack_model(model_path, input_size, hidden_layers);
  if (!model_optional.has_value()) {
    return 0;
  }
  auto &model = *model_optional;

  // get input, process it into arma::vec. This is likely to change when exp mlp
  // is trained

  // specify dimension
  auto tensor_dim_array = tensor_and_shape_jsons[1];
  if (tensor_dim_array.size() > 4) {
    // max allowed tensor dim is 4
    return 0;
  }

  // specify datatype
  int ttnn_tensor_dtype =
      tensor_and_shape_jsons[0]["tensor_spec"]["tensor_layout"]["dtype"];
  std::vector<int> onehot_dtype = get_one_hot_dtype(ttnn_tensor_dtype);

  // specify memory_config
  int mem_cfg = tensor_and_shape_jsons[0]["tensor_spec"]["tensor_layout"]
                                      ["memory_config"]["buffer_type"];
  std::vector<int> memory_config = get_memory_config(mem_cfg);

  // create input and output vectors. This is subject to change based on how
  // categorical data is encoded in model current implementation is onehot
  // encoding for datatype and memory_config
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
      static_cast<double>(memory_config[1])};
  arma::vec output(1);

  // model inference
  model.Predict(input, output);

  return output(0);
}

}//namespace op_perf