include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(EXTERNAL_LIBRARIES
    ${nlohmann_LIBRARIES}
    ${Armadillo_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

add_definitions(-DARMA_DONT_USE_WRAPPER)

enable_testing()

add_executable(
  test_interface
  test_interface.cpp
)

target_link_libraries(test_interface PUBLIC
  GTest::gtest_main
  ${EXTERNAL_LIBRARIES}
  ops_library
  interface_library
)

include(GoogleTest)
gtest_discover_tests(test_interface)