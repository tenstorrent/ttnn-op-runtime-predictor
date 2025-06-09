cmake_minimum_required(VERSION 3.30)
project(OpModeling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(Armadillo REQUIRED)
find_package(GTest REQUIRED)

if (APPLE)
    # TODO: replace "/opt/homebrew" with the output of `brew --prefix`
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
elseif (LINUX)
    find_package(mlpack 4.0.0 QUIET)
    if (NOT mlpack_FOUND)
        # Note: using default path when compiling from source
        if (EXISTS "/usr/local/include/mlpack.hpp")
            include_directories(/usr/local/include)
        else()
            message(STATUS "ERROR: mlpack could not be found")
        endif()
    endif()
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

add_executable(test_mlpregress test_mlpregress.cpp)
target_link_libraries(test_mlpregress armadillo)

enable_testing()

#add_executable(test2 test2.cc)
#target_link_libraries(test2 GTest::gtest_main)

include(GoogleTest)
#gtest_discover_tests(test2)