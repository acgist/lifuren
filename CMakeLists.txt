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
    # VSCode要和变体保持一致
    set(CMAKE_BUILD_TYPE Debug)
    # set(CMAKE_BUILD_TYPE Release)
endif()

option(ENABLE_TEST "打开测试"     ON)
option(ENABLE_FLTK "编译FLTK界面" ON)
option(ENABLE_REST "编译REST接口" ON)

option(BUILD_SHARED_LIBS "是否编译为共享库" OFF)

set(HEADER_DIR include/)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# TODO: 使用C23/C++23
# https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
if(CMAKE_HOST_WIN32)
    if(MSVC)
        # https://learn.microsoft.com/zh-cn/cpp/build/reference/compiler-options-listed-by-category
        # https://learn.microsoft.com/zh-cn/cpp/error-messages/compiler-warnings/compiler-warnings-c4000-c5999
        set(CMAKE_C_STANDARD   17)
        set(CMAKE_CXX_STANDARD 20)
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} /std:c17 /std:c++20")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c17 /std:c++20")
        set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /wd4804 /wd4819")
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
    set(CMAKE_C_STANDARD   23)
    set(CMAKE_CXX_STANDARD 23)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK    "${CMAKE_COMMAND} -E time")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -std=c2x -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers -O0 -g")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers -O0 -g")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -Wunused-variable -Wunused-parameter -Wmissing-field-initializers -O3")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wunused-variable -Wunused-parameter -Wmissing-field-initializers -O3")
    if(BUILD_SHARED_LIBS)
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -fPIC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
else()
    message(FATAL_ERROR "不支持的系统：${CMAKE_HOST_SYSTEM_NAME}")
endif()

message(STATUS "当前模式: ${CMAKE_BUILD_TYPE}")

add_definitions(-DSPDLOG_COMPILED_LIB)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
    message(STATUS "C_FLAGS  : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
    message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
else()
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
    message(STATUS "C_FLAGS  : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
    message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # ${CMAKE_SOURCE_DIR}
    # ${CMAKE_HOME_DIRECTORY}
    # ${CMAKE_CURRENT_SOURCE_DIR}
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
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

set(
    CMAKE_PREFIX_PATH
    ${CMAKE_PREFIX_PATH}
    deps/libtorch
)

if(CMAKE_HOST_WIN32)
    set(
        CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        deps/fltk
        deps/faiss
        deps/ffmpeg
        deps/opencv
        deps/spdlog
        deps/yaml-cpp
        deps/cpp-httplib
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(FLTK_FLUID_EXECUTABLE deps/fltk/tools/fltk/debug)

        file(GLOB FLTK_DLLS    "deps/fltk/debug/bin/*.dll")
        file(GLOB YAML_DLLS    "deps/yaml-cpp/debug/bin/*.dll")
        file(GLOB FAISS_DLLS   "deps/faiss/debug/bin/*.dll")
        file(GLOB FFMPEG_DLLS  "deps/ffmpeg/debug/bin/*.dll")
        file(GLOB SPDLOG_DLLS  "deps/spdlog/debug/bin/*.dll")
        file(GLOB OPENCV_DLLS  "deps/opencv/x64/vc16/bin/opencv_world4100d.dll")
        file(GLOB HTTPLIB_DLLS "deps/cpp-httplib/debug/bin/*.dll")
    else()
        set(FLTK_FLUID_EXECUTABLE deps/fltk/tools/fltk)

        file(GLOB FLTK_DLLS    "deps/fltk/bin/*.dll")
        file(GLOB YAML_DLLS    "deps/yaml-cpp/bin/*.dll")
        file(GLOB FAISS_DLLS   "deps/faiss/bin/*.dll")
        file(GLOB FFMPEG_DLLS  "deps/ffmpeg/bin/*.dll")
        file(GLOB SPDLOG_DLLS  "deps/spdlog/bin/*.dll")
        file(GLOB OPENCV_DLLS  "deps/opencv/x64/vc16/bin/opencv_world4100.dll")
        file(GLOB HTTPLIB_DLLS "deps/cpp-httplib/bin/*.dll")
    endif()

    file(GLOB TORCH_DLLS       "deps/libtorch/lib/*.dll")

    find_package(FFMPEG        REQUIRED)
    find_package(libpng        REQUIRED)
    find_package(libjpeg-turbo REQUIRED)
endif()

find_package(FLTK     REQUIRED)
find_package(faiss    REQUIRED)
find_package(Torch    REQUIRED)
find_package(OpenCV   REQUIRED)
find_package(OpenMP   REQUIRED)
find_package(spdlog   REQUIRED)
find_package(httplib  REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(cv)
add_subdirectory(nlp)
add_subdirectory(boot)
add_subdirectory(core)
add_subdirectory(deps)
add_subdirectory(model)
add_subdirectory(client)

set(HELP_FILES "LICENSE" "README.md")

install(FILES ${HELP_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/)
install(DIRECTORY docs/resources/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.png" PATTERN "*.jpg" PATTERN "*.ico" PATTERN "*.jpeg")

file(COPY ${HELP_FILES}   DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/)
file(COPY docs/resources/ DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin FILES_MATCHING PATTERN "*.png" PATTERN "*.jpg" PATTERN "*.ico" PATTERN "*.jpeg")
