// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <mlpack.hpp>
#include <nlohmann/json.hpp>

#include "interface/interface.hpp"

using namespace op_perf;

// utility function, create serialized ttnn::tensor as it is sent from
// metal-side. May change if metal-side serialization changes
nlohmann::json create_serialized_tensor(const std::vector<int> &dimensions,
                                        const int &dtype,
                                        const int &buffer_type) {
  nlohmann::json tensor_json = {
      {"tensor_spec",
       {{"logical_shape", dimensions},
        {"tensor_layout",
         {{"alignment", {{"value", {1, 1, 32, 32}}}},
          {"dtype", dtype},
          {"memory_config",
           {{"buffer_type", buffer_type},
            {"created_with_nd_shard_spec", false},
            {"memory_layout", 0}}},
          {"page_config",
           {{"config",
             {{"index", 1},
              {"value",
               {{"tile",
                 {{"face_shape", {16, 16}},
                  {"num_faces", 4},
                  {"tile_shape", {32, 32}}}}}}}}}}}}}},
      {"storage", {{"index", 1}, {"value", nlohmann::json::object()}}}};

  return tensor_json;
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
        std::make_tuple(
            "exp", create_serialized_tensor({32, 32, 0, 0}, BFLOAT8_B, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({2048, 2048, 0, 0}, BFLOAT16, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({8192, 8192, 0, 0}, FLOAT32, DRAM),
            nlohmann::json()),

        std::make_tuple("exp",
                        create_serialized_tensor({32, 32, 0, 0}, BFLOAT8_B, L1),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({2048, 2048, 0, 0}, BFLOAT16, L1),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({8192, 8192, 0, 0}, FLOAT32, L1),
            nlohmann::json()),
        // rank 2 rectangular tensors
        std::make_tuple("exp",
                        create_serialized_tensor({1, 32, 0, 0}, UINT16, DRAM),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({32, 4096, 0, 0}, UINT32, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({128, 8192, 0, 0}, BFLOAT8_B, DRAM),
            nlohmann::json()),
        std::make_tuple("exp",
                        create_serialized_tensor({32, 1, 0, 0}, UINT16, DRAM),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({4096, 32, 0, 0}, UINT32, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({8192, 128, 0, 0}, BFLOAT8_B, DRAM),
            nlohmann::json()),

        std::make_tuple("exp",
                        create_serialized_tensor({1, 32, 0, 0}, UINT16, L1),
                        nlohmann::json()),
        std::make_tuple("exp",
                        create_serialized_tensor({32, 4096, 0, 0}, UINT32, L1),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({128, 8192, 0, 0}, BFLOAT8_B, L1),
            nlohmann::json()),
        std::make_tuple("exp",
                        create_serialized_tensor({32, 1, 0, 0}, UINT16, L1),
                        nlohmann::json()),
        std::make_tuple("exp",
                        create_serialized_tensor({4096, 32, 0, 0}, UINT32, L1),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({8192, 128, 0, 0}, BFLOAT8_B, L1),
            nlohmann::json()),
        // rank 3 square tensors
        std::make_tuple(
            "exp", create_serialized_tensor({32, 32, 32, 0}, BFLOAT16, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({128, 128, 128, 0}, FLOAT32, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({896, 896, 896, 0}, UINT16, DRAM),
            nlohmann::json()),

        std::make_tuple("exp",
                        create_serialized_tensor({32, 32, 32, 0}, BFLOAT16, L1),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({128, 128, 128, 0}, FLOAT32, L1),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({896, 896, 896, 0}, UINT16, L1),
            nlohmann::json()),
        // rank 3 rectangular tensors
        std::make_tuple("exp",
                        create_serialized_tensor({1, 32, 32, 0}, UINT32, DRAM),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({32, 128, 128, 0}, BFLOAT8_B, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({128, 896, 896, 0}, BFLOAT16, DRAM),
            nlohmann::json()),
        std::make_tuple("exp",
                        create_serialized_tensor({32, 32, 1, 0}, UINT32, DRAM),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({128, 128, 32, 0}, BFLOAT8_B, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({896, 896, 128, 0}, BFLOAT16, DRAM),
            nlohmann::json()),

        std::make_tuple("exp",
                        create_serialized_tensor({1, 32, 32, 0}, UINT32, L1),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({32, 128, 128, 0}, BFLOAT8_B, L1),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({128, 896, 896, 0}, BFLOAT16, L1),
            nlohmann::json()),
        std::make_tuple("exp",
                        create_serialized_tensor({32, 32, 1, 0}, UINT32, L1),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({128, 128, 32, 0}, BFLOAT8_B, L1),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({896, 896, 128, 0}, BFLOAT16, L1),
            nlohmann::json()),
        // rank 4 square tensors
        std::make_tuple(
            "exp", create_serialized_tensor({32, 32, 32, 32}, FLOAT32, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({96, 96, 96, 96}, UINT16, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({256, 256, 256, 256}, UINT32, DRAM),
            nlohmann::json()),

        std::make_tuple("exp",
                        create_serialized_tensor({32, 32, 32, 32}, FLOAT32, L1),
                        nlohmann::json()),
        std::make_tuple("exp",
                        create_serialized_tensor({96, 96, 96, 96}, UINT16, L1),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({256, 256, 256, 256}, UINT32, L1),
            nlohmann::json()),
        // rank 4 rectangular tensors
        std::make_tuple(
            "exp", create_serialized_tensor({1, 1, 32, 32}, BFLOAT8_B, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({32, 32, 64, 64}, BFLOAT16, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({32, 32, 320, 320}, FLOAT32, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({32, 32, 1, 1}, BFLOAT8_B, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({64, 64, 32, 32}, BFLOAT16, DRAM),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({320, 320, 32, 32}, FLOAT32, DRAM),
            nlohmann::json()),

        std::make_tuple("exp",
                        create_serialized_tensor({1, 1, 32, 32}, BFLOAT8_B, L1),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({32, 32, 64, 64}, BFLOAT16, L1),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({32, 32, 320, 320}, FLOAT32, L1),
            nlohmann::json()),
        std::make_tuple("exp",
                        create_serialized_tensor({32, 32, 1, 1}, BFLOAT8_B, L1),
                        nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({64, 64, 32, 32}, BFLOAT16, L1),
            nlohmann::json()),
        std::make_tuple(
            "exp", create_serialized_tensor({320, 320, 32, 32}, FLOAT32, L1),
            nlohmann::json())));

// concatenate_heads success test cases (runtime estimate is returned, estimate
// is > 0)
class ConcatenateHeadsSuccessTest
    : public testing::TestWithParam<
          std::tuple<nlohmann::json, nlohmann::json>> {};

TEST_P(ConcatenateHeadsSuccessTest, ReturnsPositiveRuntime) {
  auto [arg1, arg2] = GetParam();
  auto runtime = predict_concatenate_heads_runtime(arg1, arg2);
  EXPECT_GT(runtime, 0) << "runtime is " << runtime;
}

// test bfloat16 and bfloat8, only DRAM memory config, 20 tests total
INSTANTIATE_TEST_SUITE_P(
    ConcatenateHeadsSuccess, ConcatenateHeadsSuccessTest,
    testing::Values(std::make_tuple(create_serialized_tensor({4, 32, 32, 32},
                                                             BFLOAT16, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({4, 32, 32, 32},
                                                             BFLOAT8_B, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({8, 64, 64, 32},
                                                             BFLOAT16, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({8, 64, 64, 32},
                                                             BFLOAT8_B, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({2, 128, 32, 32},
                                                             BFLOAT16, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({2, 128, 32, 32},
                                                             BFLOAT8_B, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({16, 32, 32, 32},
                                                             BFLOAT16, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({16, 32, 32, 32},
                                                             BFLOAT8_B, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({4, 256, 32, 32},
                                                             BFLOAT16, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({4, 256, 32, 32},
                                                             BFLOAT8_B, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({8, 32, 128, 32},
                                                             BFLOAT16, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({8, 32, 128, 32},
                                                             BFLOAT8_B, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({2, 32, 32, 128},
                                                             BFLOAT16, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({2, 32, 32, 128},
                                                             BFLOAT8_B, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({4, 64, 64, 64},
                                                             BFLOAT16, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({4, 64, 64, 64},
                                                             BFLOAT8_B, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({8, 32, 32, 64},
                                                             BFLOAT16, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({8, 32, 32, 64},
                                                             BFLOAT8_B, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({16, 16, 32, 32},
                                                             BFLOAT16, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}})),
                    std::make_tuple(create_serialized_tensor({16, 16, 32, 32},
                                                             BFLOAT8_B, DRAM),
                                    nlohmann::json({{"buffer_type", DRAM}}))));

// create_qkv_heads success test cases (runtime estimate is returned, estimate
// is > 0)
class CreateQKVHeadsSuccessTest
    : public testing::TestWithParam<
          std::tuple<nlohmann::json, int, std::optional<int>, bool>> {};

TEST_P(CreateQKVHeadsSuccessTest, ReturnsPositiveRuntime) {
  auto [tensorspec, num_heads, num_kv_heads, transpose_k_heads] = GetParam();
  auto runtime = predict_create_qkv_heads_runtime(
      tensorspec, num_heads, num_kv_heads, transpose_k_heads);
  EXPECT_GT(runtime, 0) << "runtime is " << runtime;
}

// 20 tests total
INSTANTIATE_TEST_SUITE_P(
    CreateQKVHeadsSuccess, CreateQKVHeadsSuccessTest,
    testing::Values(
        // BFLOAT16, nullopt, transpose_k_heads = false
        std::make_tuple(create_serialized_tensor({1, 1, 32, 1152}, BFLOAT16,
                                                 L1),
                        8, std::nullopt, false),
        std::make_tuple(create_serialized_tensor({2, 1, 64, 3072}, BFLOAT16,
                                                 L1),
                        16, std::nullopt, false),
        std::make_tuple(create_serialized_tensor({4, 1, 128, 4096}, BFLOAT16,
                                                 L1),
                        32, std::nullopt, false),
        std::make_tuple(create_serialized_tensor({6, 1, 256, 6144}, BFLOAT16,
                                                 L1),
                        48, std::nullopt, false),
        std::make_tuple(create_serialized_tensor({8, 1, 512, 7168}, BFLOAT16,
                                                 L1),
                        64, std::nullopt, false),

        // BFLOAT16, num_kv_heads, transpose_k_heads = true
        std::make_tuple(create_serialized_tensor({1, 1, 32, 1152}, BFLOAT16,
                                                 L1),
                        8, 8, true),
        std::make_tuple(create_serialized_tensor({2, 1, 64, 3072}, BFLOAT16,
                                                 L1),
                        16, 16, true),
        std::make_tuple(create_serialized_tensor({4, 1, 128, 4096}, BFLOAT16,
                                                 L1),
                        32, 32, true),
        std::make_tuple(create_serialized_tensor({6, 1, 256, 6144}, BFLOAT16,
                                                 L1),
                        48, 24, true),
        std::make_tuple(create_serialized_tensor({8, 1, 512, 7168}, BFLOAT16,
                                                 L1),
                        64, 32, true),

        // BFLOAT8_B, nullopt, transpose_k_heads = true
        std::make_tuple(create_serialized_tensor({1, 1, 32, 1152}, BFLOAT8_B,
                                                 L1),
                        8, std::nullopt, true),
        std::make_tuple(create_serialized_tensor({2, 1, 64, 3072}, BFLOAT8_B,
                                                 L1),
                        16, std::nullopt, true),
        std::make_tuple(create_serialized_tensor({4, 1, 128, 4096}, BFLOAT8_B,
                                                 L1),
                        32, std::nullopt, true),
        std::make_tuple(create_serialized_tensor({6, 1, 256, 6144}, BFLOAT8_B,
                                                 L1),
                        48, std::nullopt, true),
        std::make_tuple(create_serialized_tensor({8, 1, 512, 7168}, BFLOAT8_B,
                                                 L1),
                        64, std::nullopt, true),

        // BFLOAT8_B, num_kv_heads, transpose_k_heads = false
        std::make_tuple(create_serialized_tensor({1, 1, 32, 1152}, BFLOAT8_B,
                                                 L1),
                        8, 8, false),
        std::make_tuple(create_serialized_tensor({2, 1, 64, 3072}, BFLOAT8_B,
                                                 L1),
                        16, 16, false),
        std::make_tuple(create_serialized_tensor({4, 1, 128, 4096}, BFLOAT8_B,
                                                 L1),
                        32, 32, false),
        std::make_tuple(create_serialized_tensor({6, 1, 256, 6144}, BFLOAT8_B,
                                                 L1),
                        48, 24, false),
        std::make_tuple(create_serialized_tensor({8, 1, 512, 7168}, BFLOAT8_B,
                                                 L1),
                        64, 32, false)));

// paged_sdpa_decode success test cases (runtime estimate is returned, estimate
// is > 0)
class PagedSDPADecodeSuccessTest
    : public testing::TestWithParam<
          std::tuple<nlohmann::json, nlohmann::json, nlohmann::json,
                     nlohmann::json, std::optional<nlohmann::json>,
                     std::optional<nlohmann::json>, bool, float, int, int,
                     nlohmann::json, int, int, int, int, int, bool, bool>> {};

TEST_P(PagedSDPADecodeSuccessTest, ReturnsPositiveRuntime) {
  auto [q_tensor, k_tensor, v_tensor, page_table_tensor, cur_pos_tensor,
        attn_mask_tensor, is_causal, scale, k_chunk_size, q_chunk_size,
        output_memory_config, math_fidelity, math_approx_mode, fp32_dest_acc_en,
        packer_l1_acc, exp_approx_mode, use_sdpa_program_config,
        use_compute_kernel_config] = GetParam();

  auto runtime = predict_paged_sdpa_decode_runtime(
      q_tensor, k_tensor, v_tensor, page_table_tensor, cur_pos_tensor,
      attn_mask_tensor, is_causal, scale, k_chunk_size, q_chunk_size,
      output_memory_config, math_fidelity, math_approx_mode, fp32_dest_acc_en,
      packer_l1_acc, exp_approx_mode, use_sdpa_program_config,
      use_compute_kernel_config);
  EXPECT_GT(runtime, 0) << "runtime is " << runtime;
}

INSTANTIATE_TEST_SUITE_P(
    PagedSDPADecodeSuccess, PagedSDPADecodeSuccessTest,
    testing::Values(
        // BFLOAT16, is_causal=true, scale=1.0
        std::make_tuple(
            create_serialized_tensor({1, 8, 32, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 128}, BFLOAT16, L1), std::nullopt,
            std::nullopt, true, 1.0f, 64, 128,
            nlohmann::json{{"buffer_type", L1},
                           {"created_with_nd_shard_spec", false},
                           {"memory_layout", 0}},
            0 /*math_fidelity*/, true /*math_approx_mode*/,
            true /*fp32_dest_acc_en*/, false /*packer_l1_acc*/, 0, true, true),
        // BFLOAT8_B, is_causal=false, scale=0.5
        std::make_tuple(
            create_serialized_tensor({1, 16, 32, 64}, BFLOAT8_B, L1),
            create_serialized_tensor({1, 16, 1024, 64}, BFLOAT8_B, L1),
            create_serialized_tensor({1, 16, 1024, 64}, BFLOAT8_B, L1),
            create_serialized_tensor({1, 64}, BFLOAT8_B, L1), std::nullopt,
            std::nullopt, false, 0.5f, 32, 64,
            nlohmann::json{{"buffer_type", DRAM},
                           {"created_with_nd_shard_spec", false},
                           {"memory_layout", 0}},
            1 /*math_fidelity*/, false /*math_approx_mode*/,
            false /*fp32_dest_acc_en*/, true /*packer_l1_acc*/, 1, true, true),
        // With cur_pos tensor
        std::make_tuple(
            create_serialized_tensor({1, 8, 32, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 128}, BFLOAT16, L1),
            create_serialized_tensor({128}, UINT32, L1), std::nullopt, true,
            1.0f, 64, 128,
            nlohmann::json{{"buffer_type", L1},
                           {"created_with_nd_shard_spec", false},
                           {"memory_layout", 0}},
            0, true, true, false, 0, true, true),
        // With attn_mask tensor
        std::make_tuple(
            create_serialized_tensor({1, 8, 32, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 128}, BFLOAT16, L1), std::nullopt,
            create_serialized_tensor({1, 1, 32, 2048}, BFLOAT16, L1), true,
            1.0f, 64, 128,
            nlohmann::json{{"buffer_type", L1},
                           {"created_with_nd_shard_spec", false},
                           {"memory_layout", 0}},
            0, true, true, false, 0, true, true),
        // No sdpa_program_config
        std::make_tuple(
            create_serialized_tensor({1, 8, 32, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 128}, BFLOAT16, L1), std::nullopt,
            std::nullopt, true, 1.0f, 64, 128,
            nlohmann::json{{"buffer_type", L1},
                           {"created_with_nd_shard_spec", false},
                           {"memory_layout", 0}},
            0, true, true, false, 0, false, true),
        // No compute_kernel_config
        std::make_tuple(
            create_serialized_tensor({1, 8, 32, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 128}, BFLOAT16, L1), std::nullopt,
            std::nullopt, true, 1.0f, 64, 128,
            nlohmann::json{{"buffer_type", L1},
                           {"created_with_nd_shard_spec", false},
                           {"memory_layout", 0}},
            0, true, true, false, 0, true, false),
        // With all optional args
        std::make_tuple(
            create_serialized_tensor({1, 8, 32, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 8, 2048, 64}, BFLOAT16, L1),
            create_serialized_tensor({1, 128}, BFLOAT16, L1),
            create_serialized_tensor({128}, UINT32, L1),
            create_serialized_tensor({1, 1, 32, 2048}, BFLOAT16, L1), true,
            1.0f, 64, 128,
            nlohmann::json{{"buffer_type", L1},
                           {"created_with_nd_shard_spec", false},
                           {"memory_layout", 0}},
            0, true, true, false, 0, false, false)

            ));

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
        std::make_tuple("exp",
                        create_serialized_tensor({1, 2, 3, 4, 5}, FLOAT32, L1),
                        nlohmann::json()),
        std::make_tuple("exp", create_serialized_tensor({1}, UINT16, DRAM),
                        nlohmann::json()),
        std::make_tuple("exp", create_serialized_tensor({}, BFLOAT16, L1),
                        nlohmann::json()),
        // json args not properly configured
        std::make_tuple("exp", nlohmann::json(), nlohmann::json()),
        // wrong parameter type
        std::make_tuple("exp", 42, 24)));
