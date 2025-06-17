cmake_minimum_required(VERSION 3.30)
project(OpModeling LANGUAGES CXX)

SET(ROOTDIR "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${ROOTDIR}")

find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)
find_package(mlpack REQUIRED)
find_package(Armadillo REQUIRED)
find_package(nlohmann_json REQUIRED)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_definitions(-DARMA_DONT_USE_WRAPPER)

include(CTest)

add_subdirectory(ops)
add_subdirectory(tests)