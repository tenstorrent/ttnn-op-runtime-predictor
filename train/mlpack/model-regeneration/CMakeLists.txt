set(EXTERNAL_INCLUDE_DIRS
    ${nlohmann_INCLUDE_DIRS}
    ${Armadillo_INCLUDE_DIRS}
)

set(EXTERNAL_LIBRARIES
    ${nlohmann_LIBRARIES}
    ${Armadillo_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

add_executable(retrain_mlp retrain_mlp.cpp mlp_config_utils.hpp train_mlp_utils.hpp)
add_executable(train_new_mlp train_new_mlp.cpp mlp_config_utils.hpp train_mlp_utils.hpp)

target_link_libraries(retrain_mlp PRIVATE ${EXTERNAL_LIBRARIES})
target_link_libraries(train_new_mlp PRIVATE ${EXTERNAL_LIBRARIES})