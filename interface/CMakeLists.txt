add_library(interface_library)

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

target_include_directories(interface_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EXTERNAL_INCLUDE_DIRS})

file(GLOB_RECURSE INTERFACE_SRC src/*.cpp)

target_sources(interface_library PRIVATE ${INTERFACE_SRC})

target_link_libraries(interface_library PUBLIC
    ${EXTERNAL_LIBRARIES}
)

