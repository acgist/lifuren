cmake_minimum_required(VERSION 3.21.0)

project(lifuren VERSION 1.0.0 LANGUAGES C CXX)

SET(CMAKE_BUILD_TYPE Debug)
# SET(CMAKE_BUILD_TYPE Release)

# https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_C_STANDARD   17)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_HOST_UNIX)
    set(CMAKE_C_FLAGS           "-Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG} -Wattributes -Wunused-parameter -std=c17 -O0 -g")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -std=c17 -O3")
    set(CMAKE_CXX_FLAGS         "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Wattributes -Wunused-parameter -std=c++17 -O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17 -O3")
elseif(CMAKE_HOST_WIN32)
    if(!MSVC)
        message("需要安装MSVC")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
else()
    message("不支持的系统：${CMAKE_HOST_SYSTEM_NAME}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/debug/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/debug/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/debug/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/release/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/release/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/release/bin)

set(
    CMAKE_INSTALL_PREFIX
    ${CMAKE_SOURCE_DIR}/target/
)

set(
    CMAKE_INSTALL_INCLUDEDIR
    ${CMAKE_INSTALL_PREFIX}/include/
)

if(CMAKE_HOST_WIN32)
    set(
        CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        deps/dlib
        deps/fltk
        deps/mathgl
        deps/opencv
        deps/libtorch
    )

    include_directories(
        deps/dlib/include
        deps/fltk/include
        deps/mathgl/include
        deps/opencv/include
        deps/libtorch/include
        deps/libtorch/include/torch/csrc/api/include
    )

    set(FLTK_FLUID_EXECUTABLE deps/fltk/tools/fltk)
else()
    set(
        CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        deps/libtorch
    )
    include_directories(
        deps/libtorch/include
        deps/libtorch/include/torch/csrc/api/include
    )
endif()


find_package(dlib    REQUIRED)
find_package(FLTK    REQUIRED)
find_package(Torch   REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(MathGL2 REQUIRED)

if(CMAKE_HOST_WIN32)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        file(GLOB DLIB_DLLS   "deps/dlib/debug/bin/*.dll")
        file(GLOB FLTK_DLLS   "deps/fltk/debug/bin/*.dll")
        file(GLOB MATHGL_DLLS "deps/mathgl/debug/bin/*.dll")
    else()
        file(GLOB DLIB_DLLS   "deps/dlib/bin/*.dll")
        file(GLOB FLTK_DLLS   "deps/fltk/bin/*.dll")
        file(GLOB MATHGL_DLLS "deps/mathgl/bin/*.dll")
    endif()
    file(GLOB OPENCV_DLLS "deps/opencv/x64/vc16/bin/*.dll")
    file(GLOB TORCH_DLLS  "deps/libtorch/lib/*.dll")
else()
    file(GLOB TORCH_DLLS  "deps/libtorch/lib/*.dll")
endif()

add_subdirectory(deps)
add_subdirectory(gg)
add_subdirectory(hh)
add_subdirectory(xx)
