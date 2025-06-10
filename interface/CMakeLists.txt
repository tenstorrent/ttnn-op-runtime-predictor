add_library(interface_library)

target_include_directories(interface_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EXTERNAL_INCLUDE_DIRS})

file(GLOB_RECURSE INTERFACE_SRC src/*.cpp)

target_sources(interface_library PRIVATE ${INTERFACE_SRC})

target_link_libraries(interface_library PUBLIC
    nlohmann_json::nlohmann_json
    ${EXTERNAL_LIBRARIES}
)

