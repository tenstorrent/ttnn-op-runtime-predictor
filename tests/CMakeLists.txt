include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

enable_testing()

add_executable(
  test_interface
  test_interface.cpp
)

target_link_libraries(test_interface PUBLIC
  interface_library
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_interface)