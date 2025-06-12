#include <gtest/gtest.h>
#include "../interface/include/interface.hpp"
#include <mlpack.hpp>
#include <nlohmann/json.hpp>

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

}

TEST(test_interface, unknownOpNoArgs){

    std::string op_name = "ttnn::exp";
    nlohmann::json arg1;
    arg1["test"] = "hello";
    arg1["test2"] = "world";

    EXPECT_EQ(get_runtime_from_model(op_name, arg1), 1);
}

TEST(test_interface, UnknownWithArgs){

    std::string op_name = "ttnn::exp";
    nlohmann::json arg1;
    arg1["test"] = "hello";
    arg1["test2"] = "world";

    EXPECT_EQ(get_runtime_from_model(op_name, arg1), 1);
} */
