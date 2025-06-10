#include <gtest/gtest.h>
#include "../interface/include/interface.hpp"
#include <mlpack.hpp>
#include <nlohmann/json.hpp>

// Demonstrate some basic assertions.
TEST(test_interface2, initialAsserts) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);

  nlohmann::json json;
  json["test"] = 1;
  EXPECT_EQ(json["test"], 1);
  std::cout << "here" << std::endl; 
}