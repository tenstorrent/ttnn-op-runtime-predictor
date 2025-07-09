cmake_minimum_required(VERSION 3.30)
project(OpModeling LANGUAGES CXX)

if(DEFINED BUILD_MLP_OP_PERF)
   	SET(MLP_OP_PERF_ROOTDIR "${PROJECT_SOURCE_DIR}")
else()
    SET(MLP_OP_PERF_ROOTDIR "${CMAKE_SOURCE_DIR}")
endif()
INCLUDE_DIRECTORIES("${MLP_OP_PERF_ROOTDIR}")

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