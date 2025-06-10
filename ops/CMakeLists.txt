add_library(ops_library)

target_include_directories(ops_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EXTERNAL_INCLUDE_DIRS})

file(GLOB_RECURSE OPS_SRC src/*.cpp)
target_sources(ops_library PRIVATE ${OPS_SRC})

target_link_libraries(ops_library PUBLIC
    interface_library
    Armadillo
    
)