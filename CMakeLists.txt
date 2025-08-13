cmake_minimum_required(VERSION 3.30)
project(OpModeling LANGUAGES CXX)

SET(ROOTDIR "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${ROOTDIR}")

add_definitions(-DMODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/train/mlpack/")
add_definitions(-DMLP_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/train/mlpack/model-regeneration/")

find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)
find_package(Armadillo REQUIRED)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_definitions(-DARMA_DONT_USE_WRAPPER)

include(CTest)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CPM)

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

add_subdirectory(ops)
add_subdirectory(tests)
add_subdirectory(train/mlpack/model-regeneration)
