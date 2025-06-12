cmake_minimum_required(VERSION 3.30)
project(OpModeling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_PREFIX_PATH "/localdev/brapanan/mlpack-4.6.2/build")

find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)
find_package(mlpack REQUIRED)
find_package(Armadillo REQUIRED)
find_package(nlohmann_json REQUIRED)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_definitions(-DARMA_DONT_USE_WRAPPER)

include(CTest)

add_subdirectory(interface)
add_subdirectory(ops)
add_subdirectory(tests)

#add_executable(main main.cpp)
#target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/interface/include ${EXTERNAL_LIBRARIES} ${EXTERNAL_INCLUDE_DIRS})