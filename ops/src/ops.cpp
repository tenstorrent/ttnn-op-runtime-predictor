// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops.hpp"
#include "model.hpp"

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

std::vector<int> get_tensor_dimensions(const nlohmann::json &tensor_dim_array,
                                       const std::optional<int> &max_rank) {
  std::vector<int> dim_vector = tensor_dim_array;
  const int length = dim_vector.size();
  if (!max_rank.has_value()) {
    max_rank = length;
  }

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

uint64_t
predict_eltwise_unary_runtime(const std::string &op_name,
                              const nlohmann::json &tensor_json,
                              const nlohmann::json &optional_output_layout) {

  if (tensor_json.is_null() || tensor_json.is_number()) {
    return 0;
  }

  // set model parameters and model path
  const int input_size = 11;
  const std::vector<int> hidden_layers = {128, 128, 128};

  // mlp and scaler filepaths
  const std::string model_path =
      std::string(MODEL_PATH) + op_name + "_mlp_model.bin";
  const std::string scaler_path =
      std::string(MODEL_PATH) + op_name + "_scaler.bin";

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
  if (tensor_json["tensor_spec"]["logical_shape"].size() < 2 ||
      tensor_json["tensor_spec"]["logical_shape"].size() > 4) {
    // max allowed tensor dim is 4
    return 0;
  }
  std::vector<int> tensor_dim_array =
      get_tensor_dimensions(tensor_json["tensor_spec"]["logical_shape"], 4);

  // specify datatype
  int ttnn_tensor_dtype = tensor_json["tensor_spec"]["tensor_layout"]["dtype"];
  std::vector<int> onehot_dtype = get_one_hot_dtype(ttnn_tensor_dtype);

  // specify memory_config
  int mem_cfg = tensor_json["tensor_spec"]["tensor_layout"]["memory_config"]
                           ["buffer_type"];
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

uint64_t
predict_concatenate_heads_runtime(const nlohmann::json &tensor_json,
                                  const nlohmann::json &output_layout) {

  // to be completed when serialization format is finalized
  if (tensor_json.is_null() || tensor_json.is_number()) {
    return 0;
  }

  // set model parameters and model path
  const int input_size = 10;
  const std::vector<int> hidden_layers = {128, 128, 128};

  // mlp and scaler filepaths
  const std::string model_path =
      std::string(MODEL_PATH) + "concatenate_heads_mlp_model.bin";
  const std::string scaler_path =
      std::string(MODEL_PATH) + "concatenate_heads_scaler.bin";

  // load mlp
  auto model_optional =
      load_mlpack_model(model_path, input_size, hidden_layers);
  if (!model_optional.has_value()) {
    return 0;
  }

  mlpack::data::StandardScaler scaler;
  try {
    mlpack::data::Load(scaler_path, "scaler", scaler, true);
  } catch (std::exception &e) {
    return 0;
  }
  auto &model = *model_optional;

  // get input, process it into arma::vec
  // specify dimension
  if (tensor_json["tensor_spec"]["logical_shape"].size() != 4) {
    // allowed tensor dim is 4
    return 0;
  }
  nlohmann::json tensor_dim_array = tensor_json["tensor_spec"]["logical_shape"];
  std::vector<int> tensor_dim = tensor_dim_array;

  // specify datatype
  int ttnn_tensor_dtype = tensor_json["tensor_spec"]["tensor_layout"]["dtype"];
  std::vector<int> onehot_dtype = get_one_hot_dtype(ttnn_tensor_dtype);

  // specify input memory_config
  int input_mem_cfg = tensor_json["tensor_spec"]["tensor_layout"]
                                 ["memory_config"]["buffer_type"];
  std::vector<int> input_mem_cfg_vector = get_memory_config(input_mem_cfg);

  // specify output memory_config
  int output_mem_cfg = output_layout["buffer_type"];
  std::vector<int> output_mem_cfg_vector = get_memory_config(output_mem_cfg);

  // create input vector
  arma::vec input = {static_cast<double>(tensor_dim[0]),
                     static_cast<double>(tensor_dim[1]),
                     static_cast<double>(tensor_dim[2]),
                     static_cast<double>(tensor_dim[3]),
                     static_cast<double>(onehot_dtype[0]),
                     static_cast<double>(onehot_dtype[1]),
                     static_cast<double>(input_mem_cfg_vector[0]),
                     static_cast<double>(input_mem_cfg_vector[1]),
                     static_cast<double>(output_mem_cfg_vector[0]),
                     static_cast<double>(output_mem_cfg_vector[1])};

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

uint64_t predict_create_qkv_heads_runtime(
    const nlohmann::json &tensor_json, const int &num_heads,
    const std::optional<int> &num_kv_heads, const bool &transpose_k_heads) {

  if (tensor_json.is_null() || tensor_json.is_number()) {
    return 0;
  }

  // set model parameters and model path
  const int input_size = 9;
  const std::vector<int> hidden_layers = {128, 128, 128};

  // mlp and scaler filepaths
  const std::string model_path =
      std::string(MODEL_PATH) + "create_qkv_heads_mlp_model.bin";
  const std::string scaler_path =
      std::string(MODEL_PATH) + "create_qkv_heads_scaler.bin";

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
  if (tensor_json["tensor_spec"]["logical_shape"].size() != 4) {
    // max allowed tensor dim is 4
    return 0;
  }
  std::vector<int> tensor_dim_array =
      get_tensor_dimensions(tensor_json["tensor_spec"]["logical_shape"], 4);

  // specify datatype
  int ttnn_tensor_dtype = tensor_json["tensor_spec"]["tensor_layout"]["dtype"];
  std::vector<int> onehot_dtype = get_one_hot_dtype(ttnn_tensor_dtype);

  int num_kv_heads_val =
      num_kv_heads.has_value() ? num_kv_heads.value() : num_heads;

  int transpose_k_heads_int = transpose_k_heads ? 1 : 0;

  // create input vector
  arma::vec input = {

      static_cast<double>(tensor_dim_array[0]),
      static_cast<double>(tensor_dim_array[1]),
      static_cast<double>(tensor_dim_array[2]),
      static_cast<double>(tensor_dim_array[3]),
      static_cast<double>(onehot_dtype[0]),
      static_cast<double>(onehot_dtype[1]),
      static_cast<double>(num_heads),
      static_cast<double>(num_kv_heads_val),
      static_cast<double>(transpose_k_heads_int)

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

uint64_t predict_paged_sdpa_decode_runtime(
    const nlohmann::json &q_tensor_json, const nlohmann::json &k_tensor_json,
    const nlohmann::json &v_tensor_json,
    const nlohmann::json &page_table_tensor_json,
    const std::optional<nlohmann::json> &optional_cur_pos_tensor_json,
    const std::optional<nlohmann::json> &optional_attn_mask_tensor_json,
    const bool &is_causal, const float &optional_scale, const int &k_chunk_size,
    const int &input_dtype, const int &output_memory_config,
    const int &math_fidelity, const int &math_approx_mode,
    const int &fp32_dest_acc_en, const int &packer_l1_acc,
    const int &exp_approx_mode, const bool &use_) {

  // to be completed when serialization format is finalized
  if (q_tensor_json.is_null() || q_tensor_json.is_number() ||
      k_tensor_json.is_null() || k_tensor_json.is_number() ||
      v_tensor_json.is_null() || v_tensor_json.is_number() ||
      page_table_tensor_json.is_null() || page_table_tensor_json.is_number()) {
    return 0;
  }

  // set model parameters and model path
  const int input_size = 36;
  const std::vector<int> hidden_layers = {256, 128, 128};

  // mlp and scaler filepaths
  const std::string model_path =
      std::string(MODEL_PATH) + "paged_sdpa_decode_mlp_model.bin";
  const std::string scaler_path =
      std::string(MODEL_PATH) + "paged_sdpa_decode_scaler.bin";

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
  // specify q tensor dimension
  if (q_tensor_json["tensor_spec"]["logical_shape"].size() != 4) {
    // allowed tensor dim is 4
    return 0;
  }
  nlohmann::json q_tensor_dim_array =
      q_tensor_json["tensor_spec"]["logical_shape"];
  std::vector<int> q_tensor_dim = q_tensor_dim_array;

  // specify k tensor dimension
  if (k_tensor_json["tensor_spec"]["logical_shape"].size() != 4) {
    // allowed tensor dim is 4
    return 0;
  }
  nlohmann::json k_tensor_dim_array =
      k_tensor_json["tensor_spec"]["logical_shape"];
  std::vector<int> k_tensor_dim = k_tensor_dim_array;

  // specify v tensor dimension
  if (v_tensor_json["tensor_spec"]["logical_shape"].size() != 4) {
    // allowed tensor dim is 4
    return 0;
  }
  nlohmann::json v_tensor_dim_array =
      v_tensor_json["tensor_spec"]["logical_shape"];
  std::vector<int> v_tensor_dim = v_tensor_dim_array;

  // specify page table tensor dimension
  if (page_table_tensor_json["tensor_spec"]["logical_shape"].size() != 2) {
    // allowed tensor dim is 2
    return 0;
  }
  nlohmann::json page_table_tensor_dim_array =
      page_table_tensor_json["tensor_spec"]["logical_shape"];
  std::vector<int> page_table_tensor_dim = page_table_tensor_dim_array;

  // specify optional cur pos tensor dimension
  std::vector<int> cur_pos_tensor_dim = {-1};
  if (optional_cur_pos_tensor_json.has_value()) {
    if (optional_cur_pos_tensor_json.value()["tensor_spec"]["logical_shape"]
            .size() != 1) {
      // allowed tensor dim is 1
      return 0;
    }
    nlohmann::json cur_pos_tensor_dim_array =
        optional_cur_pos_tensor_json.value()["tensor_spec"]["logical_shape"];
    cur_pos_tensor_dim = cur_pos_tensor_dim_array;
  }

  // specify optional attention mask tensor dimension
  std::vector<int> attn_mask_tensor_dim = {-1, -1, -1, -1};
  if (optional_attn_mask_tensor_json.has_value()) {
    if (optional_attn_mask_tensor_json.value()["tensor_spec"]["logical_shape"]
            .size() != 4) {
      // allowed tensor dim is 4
      return 0;
    }
    nlohmann::json attn_mask_tensor_dim_array =
        optional_attn_mask_tensor_json.value()["tensor_spec"]["logical_shape"];
    attn_mask_tensor_dim = attn_mask_tensor_dim_array;
  }

  // specify datatype
  int ttnn_tensor_dtype =
      q_tensor_json["tensor_spec"]["tensor_layout"]["dtype"];
  std::vector<int> onehot_dtype = get_one_hot_dtype(ttnn_tensor_dtype);

  // specify input memory_config
  int input_mem_cfg = q_tensor_json["tensor_spec"]["tensor_layout"]
                                   ["memory_config"]["buffer_type"];
  std::vector<int> input_mem_cfg_vector = get_memory_config(input_mem_cfg);

  // specify output memory_config
  int output_mem_cfg = output_memory_config;
  std::vector<int> output_mem_cfg_vector = get_memory_config(output_mem_cfg);

  // is_causal to int
  int is_causal_int = is_causal ? 1 : 0;

  // scale is provided as float directly

  // sdpa program config params

} // namespace op_perf
