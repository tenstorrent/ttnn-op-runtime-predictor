set(EXTERNAL_INCLUDE_DIRS
    ${nlohmann_json_SOURCE_DIR}/include
    ${mlpack_SOURCE_DIR}/src
    ${Armadillo_INCLUDE_DIRS}
)

set(EXTERNAL_LIBRARIES
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

add_executable(retrain_mlp retrain_mlp.cpp mlp_config_utils.hpp train_mlp_utils.hpp)
target_include_directories(retrain_mlp PRIVATE ${EXTERNAL_INCLUDE_DIRS})
target_link_libraries(retrain_mlp ${EXTERNAL_LIBRARIES})

add_executable(train_new_mlp train_new_mlp.cpp mlp_config_utils.hpp train_mlp_utils.hpp)
target_include_directories(train_new_mlp PRIVATE ${EXTERNAL_INCLUDE_DIRS})
target_link_libraries(train_new_mlp ${EXTERNAL_LIBRARIES})

target_compile_definitions(train_new_mlp PUBLIC
    MLP_CONFIG_PATH="${ROOTDIR}/train/mlpack/model-regeneration/"
)