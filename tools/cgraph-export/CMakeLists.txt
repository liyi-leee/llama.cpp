set(TARGET cgraph-export)
get_filename_component(CMAKE_PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)


set(flatbuffers_DIR  ${CMAKE_PARENT_DIR}/install/lib/cmake/flatbuffers)  # 指向 flatbuffer 的 CMake 配置文件所在目录
message(STATUS "Find flatbuffers library in path: ${flatbuffers_DIR}")
find_package(flatbuffers CONFIG REQUIRED)

add_library(${TARGET} SHARED cgraph-export.cpp)
target_link_libraries(${TARGET} PUBLIC flatbuffers::flatbuffers)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
message(${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/ggml/include/)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/ggml/src/)


# install rules and package config for find_package(cgraph-export CONFIG)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CGRAPH_EXPORT_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of cgraph-export headers")
set(CGRAPH_EXPORT_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of cgraph-export libraries")

set_target_properties(${TARGET} PROPERTIES
    PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/cgraph-export.h;${CMAKE_CURRENT_SOURCE_DIR}/cgraph_generated.h")

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
    
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.h"
        DESTINATION include)

    # Generate and install package config and version files
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cgraph-export-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/cgraph-export-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgraph-export
        PATH_VARS CGRAPH_EXPORT_INCLUDE_INSTALL_DIR CGRAPH_EXPORT_LIB_INSTALL_DIR
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/cgraph-export-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgraph-export
    )
endif()
