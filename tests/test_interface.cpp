#include <gtest/gtest.h>
#include "../interface/include/interface.hpp"
#include <mlpack.hpp>
#include <nlohmann/json.hpp>

TEST(sanity_check, sanityCheck1) {
  
  EXPECT_STRNE("hello", "world");
  
  EXPECT_EQ(7 * 6, 42);

  nlohmann::json json;
  json["test"] = 1;
  EXPECT_EQ(json["test"], 1);
}

TEST(sanity_check, sanityCheck2) {
  
  EXPECT_STRNE("hello", "world");
  
  EXPECT_EQ(7 * 6, 42);

  nlohmann::json json;
  json["test"] = 1;
  EXPECT_EQ(json["test"], 1);
}

TEST(test_interface, isExpNoArgs){

    std::string op_name = "ttnn::exp";

    EXPECT_EQ(get_runtime_from_model(op_name), 1);
}

TEST(test_interface, isExpWithArgs){

    std::string op_name = "ttnn::exp";
    nlohmann::json arg1;
    arg1["test"] = "hello";
    arg1["test2"] = "world";

    EXPECT_EQ(get_runtime_from_model(op_name, arg1), 1);
}

/* //value parameterized tests
class GetRunTimeFromModelTest :: public testing::TestWithParam<std::tuple<std::string, nlohmann::json, nlohmann::json>> {
    protected:
        std::string op_name;
        nlohmann::json arg1;
        nlohmann::json arg2;
        nlohmann::json arg3;
}

TEST_P(GetRunTimeFromModelTest, notKnownOp){

    op_name = std::get<0>(GetParam());
    arg1 = std::get<1>(GetParam());

    ASSERT_EQ()

} */
