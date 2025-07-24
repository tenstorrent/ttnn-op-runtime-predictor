add_library(ops_library)

set(EXTERNAL_INCLUDE_DIRS
    ${Armadillo_INCLUDE_DIRS}
)

set(EXTERNAL_LIBRARIES
    ${Armadillo_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

target_include_directories(ops_library PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${mlpack_SOURCE_DIR}/src
)

file(GLOB_RECURSE OPS_SRC src/*.cpp)
target_sources(ops_library PRIVATE ${OPS_SRC})

target_link_libraries(ops_library PUBLIC
    ${EXTERNAL_LIBRARIES}
    nlohmann_json::nlohmann_json
)
