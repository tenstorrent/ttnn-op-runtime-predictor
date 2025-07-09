add_library(ops_library)

message("nlohmann directories: ${nlohmann_INCLUDE_DIRS}")
message("arma directories: ${Armadillo_INCLUDE_DIRS}")

if(DEFINED BUILD_MLP_OP_PERF)
    target_include_directories(ops_library
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    file(GLOB_RECURSE OPS_SRC src/*.cpp)
    target_sources(ops_library PRIVATE ${OPS_SRC})

    target_link_libraries(ops_library
        PUBLIC
            mlpack
            Armadillo
            nlohmann_json::nlohmann_json
            LAPACK
            BLAS
    )
else()
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

    target_include_directories(ops_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EXTERNAL_INCLUDE_DIRS})

    file(GLOB_RECURSE OPS_SRC src/*.cpp)
    target_sources(ops_library PRIVATE ${OPS_SRC})

    target_link_libraries(ops_library PUBLIC
        ${EXTERNAL_LIBRARIES}
    )
endif()