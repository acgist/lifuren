cmake_minimum_required(VERSION 3.21.0)

project(lifuren-project VERSION 1.0.0 LANGUAGES C CXX)

# -D CMAKE_BUILD_TYPE=Debug|Release
SET(CMAKE_BUILD_TYPE Debug)
# SET(CMAKE_BUILD_TYPE Release)

# https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_C_STANDARD   17)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_HOST_UNIX)
    set(CMAKE_C_FLAGS           "-std=c17 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")
    set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG} -Wunused-variable -Wunused-parameter -std=c17 -O0 -g")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -Wunused-variable -Wunused-parameter -std=c17 -O3")
    set(CMAKE_CXX_FLAGS         "-std=c++17 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Wunused-variable -Wunused-parameter -std=c++17 -O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wunused-variable -Wunused-parameter -std=c++17 -O3")
elseif(CMAKE_HOST_WIN32)
    if(MSVC)
        # /wd4804 = 忽略告警
        # /wd4819 = 忽略告警
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /wd4804 /wd4819")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4804 /wd4819")
        add_compile_options("/utf-8")
        add_compile_options("/Zc:__cplusplus")
    else()
        message("需要安装MSVC")
    endif()
else()
    message("不支持的系统：${CMAKE_HOST_SYSTEM_NAME}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/debug/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/debug/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/debug/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/release/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/release/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/release/bin)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/debug/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/debug/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/debug/bin)
else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/release/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/release/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/release/bin)
endif()

set(HEADER_DIR src/header/)
set(SOURCE_DIR src/source/)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} deps/libtorch)

set(CMAKE_INSTALL_PREFIX      ${CMAKE_SOURCE_DIR}/target/)
set(CMAKE_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/)

include_directories(
    deps/libtorch/include
    deps/libtorch/include/torch/csrc/api/include
)

if(CMAKE_HOST_WIN32)
    set(
        CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        deps/fltk
        deps/mlpack
        deps/opencv
    )

    include_directories(
        deps/fltk/include
        deps/mlpack/include
        deps/opencv/include
    )

    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(FLTK_FLUID_EXECUTABLE deps/fltk/tools/fltk/debug)
        file(GLOB FLTK_DLLS   "deps/fltk/debug/bin/*.dll")
        file(GLOB MLPACK_DLLS "deps/mlpack/debug/bin/*.dll")
    else()
        set(FLTK_FLUID_EXECUTABLE deps/fltk/tools/fltk)
        file(GLOB FLTK_DLLS   "deps/fltk/bin/*.dll")
        file(GLOB MLPACK_DLLS "deps/mlpack/bin/*.dll")
    endif()

    file(GLOB TORCH_DLLS  "deps/libtorch/lib/*.dll")
    file(GLOB OPENCV_DLLS "deps/opencv/x64/vc16/bin/*.dll")

    find_package(libpng REQUIRED)
endif()

find_package(FLTK      REQUIRED)
find_package(Torch     REQUIRED)
find_package(OpenCV    REQUIRED)
find_package(LAPACK    REQUIRED)
find_package(Armadillo REQUIRED)

add_subdirectory(deps)
add_subdirectory(core)
add_subdirectory(boot)

set(HELP "LICENSE" "README.md")

install(FILES ${HELP} DESTINATION ${CMAKE_INSTALL_PREFIX}/)

install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/data/)
install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/logs/)
install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/model/)
