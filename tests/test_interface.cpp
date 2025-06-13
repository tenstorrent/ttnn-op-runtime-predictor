#include <gtest/gtest.h>
#include "../interface/include/interface.hpp"
#include <mlpack.hpp>
#include <nlohmann/json.hpp>

//tests for get_runtime_from_model_test()

class GetRunTimeFromModelTestIsExpNoArgs : public testing::TestWithParam<std::string> {};

TEST_P(GetRunTimeFromModelTestIsExpNoArgs, isExp){

  std::string op_name = GetParam();

  if(op_name == "ttnn::exp"){
    EXPECT_EQ(get_runtime_from_model_test(op_name), 1);
  }else{
    EXPECT_NE(get_runtime_from_model_test(op_name), 1);
  }
}

INSTANTIATE_TEST_SUITE_P(GetRunTimeFromModelTestIsExpNoArgsSuite,
                         GetRunTimeFromModelTestIsExpNoArgs,
                         testing::Values("ttnn::exp", "ttnn::exp2", "failure", ""));