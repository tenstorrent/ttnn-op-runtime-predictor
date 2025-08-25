add_library(interface_library)

set(EXTERNAL_INCLUDE_DIRS
    ${Armadillo_INCLUDE_DIRS}
)

set(EXTERNAL_LIBRARIES
    ${Armadillo_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

target_include_directories(interface_library PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${mlpack_SOURCE_DIR}/src
)

file(GLOB_RECURSE INTERFACE_SRC *.cpp)
target_sources(interface_library PRIVATE ${INTERFACE_SRC})

target_link_libraries(interface_library PUBLIC
    ${EXTERNAL_LIBRARIES}
    nlohmann_json::nlohmann_json
)
