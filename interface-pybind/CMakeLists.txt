pybind11_add_module(ttnn_op_runtime_predictor
    get_runtime_from_model_pybind.cpp
)

target_include_directories(ttnn_op_runtime_predictor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${pybind11_INCLUDE_DIRS}
    ${ROOTDIR}/interface
    ${mlpack_SOURCE_DIR}/src
)

target_link_libraries(ttnn_op_runtime_predictor PRIVATE
    interface_library
    ops_library
    pybind11_json
)

install(TARGETS ttnn_op_runtime_predictor
    LIBRARY DESTINATION .
)