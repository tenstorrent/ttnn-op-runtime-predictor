// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <mlpack.hpp>
#include <nlohmann/json.hpp>

#include "../interface/interface.hpp"

using namespace op_perf;

// utility function, create serialized ttnn::tensor as it is sent from
// metal-side. May change if metal-side serialization changes
nlohmann::json create_serialized_tensor(const std::vector<int> &dimensions,
                                        const int &dtype,
                                        const int &buffer_type) {

  // two json objects are included. First is the tensor, which cannot serialize
  // tensor dims. The second object is the tensor dims.
  nlohmann::json tensor_json = {
      {"storage", {{"index", 0}, {"value", nlohmann::json::object()}}},
      {"tensor_spec",
       {{"logical_shape",
         {{"value", "tt::stl::json::to_json_t: Unsupported type "
                    "SmallVector<unsigned int>"}}},
        {"tensor_layout",
         {{"alignment",
           {{"value", "tt::stl::json::to_json_t: Unsupported type "
                      "SmallVector<unsigned int>"}}},
          {"dtype", dtype},
          {"memory_config",
           {{"buffer_type", buffer_type}, {"memory_layout", 0}}},
          {"page_config",
           {{"config",
             {{"index", 0},
              {"value",
               {{"tile",
                 {{"face_shape", {16, 16}},
                  {"num_faces", 4},
                  {"tile_shape", {32, 32}}}}}}}}}}}}}}};

  nlohmann::json shape_json = dimensions;

  return nlohmann::json::array({tensor_json, shape_json});
}

// exp success test cases (runtime estimate is returned, estimate is > 0)
class ExpSuccessTest
    : public testing::TestWithParam<
          std::tuple<std::string, nlohmann::json, nlohmann::json>> {};

TEST_P(ExpSuccessTest, ReturnsPositiveRuntime) {
  auto [op_name, arg1, arg2] = GetParam();
  auto runtime = get_runtime_from_model(op_name, arg1, arg2);
  EXPECT_GT(runtime, 0) << "runtime is " << runtime;
}

INSTANTIATE_TEST_SUITE_P(
    ExpSuccess, ExpSuccessTest,
    testing::Values(
        // rank 2 square tensors
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 0, 0}, BFLOAT8_B,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({2048, 2048, 0, 0}, BFLOAT16,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({8192, 8192, 0, 0}, FLOAT32,
                                                 DRAM),
                        nlohmann::json()),

        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 0, 0}, BFLOAT8_B, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({2048, 2048, 0, 0}, BFLOAT16,
                                                 L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({8192, 8192, 0, 0}, FLOAT32,
                                                 L1),
                        nlohmann::json()),
        // rank 2 rectangular tensors
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({1, 32, 0, 0}, UINT16, DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 4096, 0, 0}, UINT32,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({128, 8192, 0, 0}, BFLOAT8_B,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 1, 0, 0}, UINT16, DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({4096, 32, 0, 0}, UINT32,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({8192, 128, 0, 0}, BFLOAT8_B,
                                                 DRAM),
                        nlohmann::json()),

        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({1, 32, 0, 0}, UINT16, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 4096, 0, 0}, UINT32, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({128, 8192, 0, 0}, BFLOAT8_B,
                                                 L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 1, 0, 0}, UINT16, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({4096, 32, 0, 0}, UINT32, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({8192, 128, 0, 0}, BFLOAT8_B,
                                                 L1),
                        nlohmann::json()),
        // rank 3 square tensors
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 32, 0}, BFLOAT16,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({128, 128, 128, 0}, FLOAT32,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({896, 896, 896, 0}, UINT16,
                                                 DRAM),
                        nlohmann::json()),

        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 32, 0}, BFLOAT16, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({128, 128, 128, 0}, FLOAT32,
                                                 L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({896, 896, 896, 0}, UINT16,
                                                 L1),
                        nlohmann::json()),
        // rank 3 rectangular tensors
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({1, 32, 32, 0}, UINT32, DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 128, 128, 0}, BFLOAT8_B,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({128, 896, 896, 0}, BFLOAT16,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 1, 0}, UINT32, DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({128, 128, 32, 0}, BFLOAT8_B,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({896, 896, 128, 0}, BFLOAT16,
                                                 DRAM),
                        nlohmann::json()),

        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({1, 32, 32, 0}, UINT32, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 128, 128, 0}, BFLOAT8_B,
                                                 L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({128, 896, 896, 0}, BFLOAT16,
                                                 L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 1, 0}, UINT32, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({128, 128, 32, 0}, BFLOAT8_B,
                                                 L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({896, 896, 128, 0}, BFLOAT16,
                                                 L1),
                        nlohmann::json()),
        // rank 4 square tensors
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 32, 32}, FLOAT32,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({96, 96, 96, 96}, UINT16,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({256, 256, 256, 256}, UINT32,
                                                 DRAM),
                        nlohmann::json()),

        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 32, 32}, FLOAT32, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({96, 96, 96, 96}, UINT16, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({256, 256, 256, 256}, UINT32,
                                                 L1),
                        nlohmann::json()),
        // rank 4 rectangular tensors
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({1, 1, 32, 32}, BFLOAT8_B,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 64, 64}, BFLOAT16,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 320, 320}, FLOAT32,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 1, 1}, BFLOAT8_B,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({64, 64, 32, 32}, BFLOAT16,
                                                 DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({320, 320, 32, 32}, FLOAT32,
                                                 DRAM),
                        nlohmann::json()),

        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({1, 1, 32, 32}, BFLOAT8_B, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 64, 64}, BFLOAT16,
                                                 L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 320, 320}, FLOAT32,
                                                 L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({32, 32, 1, 1}, BFLOAT8_B, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({64, 64, 32, 32}, BFLOAT16,
                                                 L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({320, 320, 32, 32}, FLOAT32,
                                                 L1),
                        nlohmann::json())));

// exp input parameter validation test cases
class ExpInvalidInputTest
    : public testing::TestWithParam<
          std::tuple<std::string, nlohmann::json, nlohmann::json>> {};

TEST_P(ExpInvalidInputTest, exp_input_invalid) {
  auto [op_name, arg1, arg2] = GetParam();
  auto runtime = get_runtime_from_model(op_name, arg1, arg2);
  EXPECT_EQ(runtime, 0) << "runtime is " << runtime;
}

INSTANTIATE_TEST_SUITE_P(
    ExpInputValidation, ExpInvalidInputTest,
    testing::Values(
        // not exp
        std::make_tuple("wrong_op_name",
                        create_serialized_tensor({4, 32, 32, 32}, BFLOAT16,
                                                 DRAM),
                        nlohmann::json()),
        // wrong tensor dimension input
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({1, 2, 3, 4, 5}, FLOAT32, L1),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp",
                        create_serialized_tensor({1}, UINT16, DRAM),
                        nlohmann::json()),
        std::make_tuple("ttnn::exp", create_serialized_tensor({}, BFLOAT16, L1),
                        nlohmann::json()),
        // json args not properly configured
        std::make_tuple("ttnn::exp", nlohmann::json(), nlohmann::json()),
        // wrong parameter type
        std::make_tuple("ttnn::exp", 42, 24)));
