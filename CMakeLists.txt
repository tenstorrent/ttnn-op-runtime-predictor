cmake_minimum_required(VERSION 3.30)
project(OpModeling LANGUAGES CXX)

add_subdirectory(third_party)

include_directories("${CMAKE_BINARY_DIR}/include")

SET(ROOTDIR "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${ROOTDIR}")

add_definitions(-DMODEL_PATH="${CMAKE_SOURCE_DIR}/train/mlpack/")

find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)
find_package(Armadillo REQUIRED)
find_package(nlohmann_json REQUIRED)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_definitions(-DARMA_DONT_USE_WRAPPER)

include(CTest)

add_subdirectory(ops)
add_subdirectory(tests)

add_dependencies(ops_library mlpack)

add_dependencies(test_interface mlpack)
