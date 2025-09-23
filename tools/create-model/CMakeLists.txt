set(TARGET create-model)
get_filename_component(CMAKE_PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

add_executable(${TARGET} create-model.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)
message(${CMAKE_SOURCE_DIR})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
