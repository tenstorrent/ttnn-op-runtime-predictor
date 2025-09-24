add_library(interface_library STATIC)
set_target_properties(interface_library PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(interface_library PUBLIC
    OP_CATEGORY_PATH="${CMAKE_CURRENT_SOURCE_DIR}/op_categories/"
)

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
    ${ROOTDIR}/ops/include
)

file(GLOB_RECURSE INTERFACE_SRC *.cpp)
target_sources(interface_library PRIVATE ${INTERFACE_SRC})

target_link_libraries(interface_library PUBLIC
    ${EXTERNAL_LIBRARIES}
    nlohmann_json::nlohmann_json
)
