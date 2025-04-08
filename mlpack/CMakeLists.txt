cmake_minimum_required(VERSION 3.30)
project(OpModeling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(Armadillo REQUIRED)

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

add_executable(test_mlpregress test_mlpregress.cpp)
target_link_libraries(test_mlpregress armadillo)
