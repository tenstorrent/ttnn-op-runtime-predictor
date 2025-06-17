#include "interface/interface.hpp"
#include <gtest/gtest.h>
#include <mlpack.hpp>
#include <nlohmann/json.hpp>

// tests for get_runtime_from_model_test()

class GetRuntimeFromModelIsExpNoArgs
    : public testing::TestWithParam<std::string> {};

TEST_P(GetRuntimeFromModelIsExpNoArgs, isExp) {

  std::string op_name = GetParam();

  if (op_name == "ttnn::exp") {
    EXPECT_EQ(get_runtime_from_model(op_name), 1);
  } else {
    EXPECT_NE(get_runtime_from_model(op_name), 1);
  }
}

INSTANTIATE_TEST_SUITE_P(GetRuntimeFromModelIsExpNoArgsSuite,
                         GetRuntimeFromModelIsExpNoArgs,
                         testing::Values("ttnn::exp", "ttnn::exp2", "failure",
                                         ""));