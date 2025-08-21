#[[
Copyright(c) 2024-present acgist. All Rights Reserved.

http://www.apache.org/licenses/LICENSE-2.0

gitee : https://gitee.com/acgist/lifuren
github: https://github.com/acgist/lifuren

李夫人
北方有佳人，绝世而独立。
一顾倾人城，再顾倾人国。
宁不知倾城与倾国，佳人难再得。
]]

# https://cmake.org/cmake/help/latest/
cmake_minimum_required(VERSION 3.22.0)

project(lifuren-project VERSION 1.0.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "当前模式: ${CMAKE_BUILD_TYPE}")
message(STATUS "当前编译器: ${CMAKE_CXX_COMPILER_ID}")

option(ENABLE_TEST "打开测试" ON)

option(BUILD_SHARED_LIBS "是否编译为共享库" OFF)

set(HEADER_DIR include/)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
if(CMAKE_HOST_WIN32)
    if(MSVC)
        # https://learn.microsoft.com/zh-cn/cpp/build/reference/compiler-options-listed-by-category
        # https://learn.microsoft.com/zh-cn/cpp/error-messages/compiler-warnings/compiler-warnings-c4000-c5999
        # set(CMAKE_CXX_STANDARD 20)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /wd4804 /wd4819")
        add_compile_options("/utf-8")
        add_compile_options("/Zc:__cplusplus")
    else()
        message(FATAL_ERROR "需要安装MSVC")
    endif()
elseif(CMAKE_HOST_UNIX)
    # https://gcc.gnu.org/releases.html
    # https://gcc.gnu.org/projects/cxx-status.html
    # https://gcc.gnu.org/onlinedocs/gcc/Standards.html
    # https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html
    # set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers -O0 -g")
    # -Wunused-variable -Wno-unused-variable
    if(BUILD_SHARED_LIBS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK    "${CMAKE_COMMAND} -E time")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
else()
    message(FATAL_ERROR "不支持的系统：${CMAKE_HOST_SYSTEM_NAME}")
endif()

add_definitions(-DSPDLOG_COMPILED_LIB)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
    message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")

    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
else()
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
    message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} deps/libtorch)

if(CMAKE_HOST_WIN32)
    set(
        CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        deps/opencv
        deps/spdlog
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        file(GLOB OPENCV_DLLS "deps/opencv/debug/bin/*.dll")
        file(GLOB SPDLOG_DLLS "deps/spdlog/debug/bin/*.dll")
    else()
        file(GLOB OPENCV_DLLS "deps/opencv/bin/*.dll")
        file(GLOB SPDLOG_DLLS "deps/spdlog/bin/*.dll")
    endif()

    file(GLOB TORCH_DLLS "deps/libtorch/lib/*.dll")
endif()

find_package(CUDA)

if(CUDA_FOUND)
    message(STATUS "使用GPU推理")
else()
    message(STATUS "使用CPU推理")
endif()

find_package(OpenCV CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Torch  CONFIG REQUIRED) # Torch需要排在OpenCV后面

add_subdirectory(core)
add_subdirectory(cli)
if(ENABLE_TEST)
    enable_testing()
    message(STATUS "编译测试")
    add_subdirectory(test)
else()
    message(STATUS "忽略测试")
endif()

set(INSTALL_DOCUMENT "LICENSE" "README.md")
set(INSTALL_RESOURCE "docs/resources/")

install(FILES     ${INSTALL_DOCUMENT} DESTINATION ${CMAKE_INSTALL_PREFIX}/)
install(DIRECTORY ${INSTALL_RESOURCE} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin  FILES_MATCHING PATTERN "*.png" PATTERN "*.jpg" PATTERN "*.ico" PATTERN "*.jpeg")
install(DIRECTORY ${INSTALL_RESOURCE} DESTINATION ${CMAKE_INSTALL_PREFIX}/logs FILES_MATCHING PATTERN "*.log")

file(COPY ${INSTALL_DOCUMENT} DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/)
file(COPY ${INSTALL_RESOURCE} DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin  FILES_MATCHING PATTERN "*.png" PATTERN "*.jpg" PATTERN "*.ico" PATTERN "*.jpeg")
file(COPY ${INSTALL_RESOURCE} DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/logs FILES_MATCHING PATTERN "*.log")
