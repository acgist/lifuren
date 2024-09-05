cmake_minimum_required(VERSION 3.20.0)

# https://cmake.org/cmake/help/latest/

project(lifuren-project VERSION 1.0.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(ENABLE_TEST       "打开测试"     ON)
option(ENABLE_FLTK       "开启FLTK界面" ON)
option(ENABLE_REST       "开启REST接口" ON)

option(SD_CUBLAS         "是否编译CUDA依赖" OFF)
option(GGML_CUDA         "是否编译CUDA依赖" OFF)
option(JSON_BuildTests   "是否编译JSON测试" OFF)

option(BUILD_SHARED_LIBS "是否编译为共享库" OFF)

message(STATUS "当前模式: ${CMAKE_BUILD_TYPE}")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
if(CMAKE_HOST_UNIX)
    set(CMAKE_C_STANDARD   23)
    set(CMAKE_CXX_STANDARD 23)
    # https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html
    set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS}         -std=c2x -Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG}   -Wno-unused-variable -Wno-unused-parameter -O0 -g")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -Wunused-variable    -Wunused-parameter    -O3")
    set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}         -std=c++23 -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -Wno-unused-variable -Wno-unused-parameter -O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wunused-variable    -Wunused-parameter    -O3")
elseif(CMAKE_HOST_WIN32)
    if(MSVC)
        set(CMAKE_C_STANDARD   17)
        set(CMAKE_CXX_STANDARD 20)
        # https://learn.microsoft.com/zh-cn/cpp/build/reference/compiler-options-listed-by-category
        # /wd4804 = 忽略告警
        # /wd4819 = 忽略告警
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /std:c17 /std:c++20 /wd4804 /wd4819")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c17 /std:c++20 /wd4804 /wd4819")
        # add_compile_options("/J")
        add_compile_options("/utf-8")
        # add_compile_options("/Zc:char8_t")
        add_compile_options("/Zc:__cplusplus")
    else()
        message(WARNING "需要安装MSVC")
    endif()
else()
    message(FATAL_ERROR "不支持的系统：${CMAKE_HOST_SYSTEM_NAME}")
endif()

add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

set(HEADER_DIR src/header/)
set(SOURCE_DIR src/source/)

if(CMAKE_HOST_WIN32)
    set(
        CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        deps/fltk
        deps/faiss
        deps/opencv
        deps/spdlog
        deps/yaml-cpp
        deps/cpp-httplib
    )

    include_directories(
        deps/fltk/include
        deps/faiss/include
        deps/opencv/include
        deps/spdlog/include
        deps/yaml-cpp/include
        deps/cpp-httplib/include
    )

    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(FLTK_FLUID_EXECUTABLE deps/fltk/tools/fltk/debug)

        file(GLOB FLTK_DLLS    "deps/fltk/debug/bin/*.dll")
        file(GLOB YAML_DLLS    "deps/yaml-cpp/debug/bin/*.dll")
        file(GLOB FAISS_DLLS   "deps/faiss/debug/bin/*.dll")
        file(GLOB SPDLOG_DLLS  "deps/spdlog/debug/bin/*.dll")
        file(GLOB OPENCV_DLLS  "deps/opencv/x64/vc16/bin/opencv_world4100d.dll")
        file(GLOB HTTPLIB_DLLS "deps/cpp-httplib/debug/bin/*.dll")
    else()
        set(FLTK_FLUID_EXECUTABLE deps/fltk/tools/fltk)

        file(GLOB FLTK_DLLS    "deps/fltk/bin/*.dll")
        file(GLOB YAML_DLLS    "deps/yaml-cpp/bin/*.dll")
        file(GLOB FAISS_DLLS   "deps/faiss/bin/*.dll")
        file(GLOB SPDLOG_DLLS  "deps/spdlog/bin/*.dll")
        file(GLOB OPENCV_DLLS  "deps/opencv/x64/vc16/bin/opencv_world4100.dll")
        file(GLOB HTTPLIB_DLLS "deps/cpp-httplib/bin/*.dll")
    endif()

    find_package(libpng        REQUIRED)
    find_package(libjpeg-turbo REQUIRED)
endif()

find_package(FLTK     REQUIRED)
find_package(faiss    REQUIRED)
find_package(OpenCV   REQUIRED)
find_package(spdlog   REQUIRED)
find_package(httplib  REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(deps)
add_subdirectory(core)
add_subdirectory(model)
add_subdirectory(client)
add_subdirectory(cv)
add_subdirectory(nlp)
add_subdirectory(boot)

set(HELP "LICENSE" "README.md")

install(FILES ${HELP} DESTINATION ${CMAKE_INSTALL_PREFIX}/)

install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/data/)
install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/logs/)
install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/model/)
