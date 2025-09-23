set(TARGET cgraph-export)
get_filename_component(CMAKE_PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

include_directories(${CMAKE_PARENT_DIR}/flatbuffers/include)
link_directories(${CMAKE_PARENT_DIR}/flatbuffers)
add_executable(${TARGET} cgraph-export.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${TARGET} PRIVATE flatbuffers)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
message(${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src/)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
