cmake_minimum_required(VERSION 3.0.0)

project(lifurenpytorch VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_FLAGS "-Wall -Wextra -Wunused-parameter")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -std=c17 -O0 -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -std=c17 -O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wunused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++17 -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17 -O3")

if(CMAKE_HOST_UNIX)
    # link_directories()
    # include_directories()
elseif(CMAKE_HOST_WIN32)
    link_directories(
        D:/develop/libtorch/lib
    )
    include_directories(
        D:/develop/libtorch/include
        D:/develop/libtorch/include/torch/csrc/api/include
    )
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} D:/develop/libtorch)
elseif(CMAKE_HOST_APPLE)
    # link_directories()
    # include_directories()
else()
    # message("不支持的系统")
endif()

find_package(Torch REQUIRED)
# find_package(OpenCV REQUIRED)

set(WITH_GTEST  OFF)
set(WITH_GFLAGS OFF)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

enable_testing()
add_subdirectory(test)
add_subdirectory(../../deps/glog glog)

set(HEADER_DIR src/header/)
set(SOURCE_DIR src/source/)

aux_source_directory(${SOURCE_DIR} SOURCE_FILES)

add_executable(${PROJECT_NAME} src/Main.cpp ${SOURCE_FILES})

set(
    HEADER_FILES
    ${HEADER_DIR}/Glog.hpp
)

install(
    FILES
    ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    glog::glog
    -lc10
    -ltorch
    -ltorch_cpu
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HEADER_DIR}
)

# if(MSVC)
#     file(GLOB TORCH_DLLS "D:/develop/libtorch/lib/*.dll")
#     add_custom_command(
#         TARGET ${PROJECT_NAME}
#         POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#         ${TORCH_DLLS}
#         $<TARGET_FILE_DIR:${PROJECT_NAME}>
#     )
# endif(MSVC)
