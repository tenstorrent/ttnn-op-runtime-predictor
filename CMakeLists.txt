cmake_minimum_required(VERSION 3.30)
project(OpModeling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)
find_package(Armadillo REQUIRED)
find_package(nlohmann_json REQUIRED)

include(CTest)

pkg_check_modules(MLPACK REQUIRED mlpack)
pkg_check_modules(NLOHMANN REQUIRED nlohmann_json)

set(EXTERNAL_INCLUDE_DIRS
    ${MLPACK_INCLUDE_DIRS}
    ${NLOHMANN_INCLUDE_DIRS}
)

set(EXTERNAL_LIBRARIES
    ${MLPACK_LIBRARIES}
    ${NLOHMANN_LIBRARIES}
)

add_subdirectory(interface)
add_subdirectory(ops)
add_subdirectory(tests)

add_executable(main main.cpp)
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EXTERNAL_LIBRARIES} ${EXTERNAL_INCLUDE_DIRS})