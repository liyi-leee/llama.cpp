set(TARGET cgraph-export-test)
get_filename_component(CMAKE_PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

add_executable(${TARGET} ${TARGET}.cpp)
target_link_libraries(${TARGET} PRIVATE common llama cgraph-export ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)
message(${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/tools/cgraph-export)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
