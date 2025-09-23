cmake_minimum_required(VERSION 3.30)
project(OpModeling LANGUAGES CXX)

SET(ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${ROOTDIR}")

find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)
find_package(Armadillo REQUIRED)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_definitions(-DARMA_DONT_USE_WRAPPER)

include(CTest)
include(cmake/CPM.cmake)

CPMAddPackage(
    NAME nlohmann_json
    GITHUB_REPOSITORY nlohmann/json
    GIT_TAG v3.11.3
    OPTIONS
    "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
    "JSON_Install ON"
    "JSON_BuildTests OFF"
)

CPMAddPackage(
  NAME mlpack
  GITHUB_REPOSITORY mlpack/mlpack
  GIT_TAG 8fc23ba711075402c9cd77cfa0938a9d33256ca2
  OPTIONS
    "BUILD_GO_SHLIB OFF"
    "BUILD_CLI_EXECUTABLES OFF"
)

CPMAddPackage(
  NAME pybind11
  GITHUB_REPOSITORY pybind/pybind11
  GIT_TAG v2.13.6
  OPTIONS 
    "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
)

add_subdirectory(interface)
add_subdirectory(ops)
add_subdirectory(tests)
add_subdirectory(train/mlpack/model-regeneration)
add_subdirectory(scripts)
