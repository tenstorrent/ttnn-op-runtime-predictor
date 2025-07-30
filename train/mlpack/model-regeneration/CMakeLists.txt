add_library(model_regen_library)

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

target_include_directories(model_regen_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EXTERNAL_INCLUDE_DIRS})

file(GLOB_RECURSE MODEL_REGEN_SRC *.cpp)
target_sources(model_regen_library PRIVATE ${MODEL_REGEN_SRC})

target_link_libraries(model_regen_library PUBLIC
    ${EXTERNAL_LIBRARIES}
)

add_executable(retrain_mlp retrain_mlp.cpp)
target_link_libraries(retrain_mlp model_regen_library)

add_executable(train_new_mlp train_new_mlp.cpp)
target_link_libraries(train_new_mlp model_regen_library)