cmake_minimum_required(VERSION 3.0.0)

project(lifuren VERSION 1.0.0 LANGUAGES C CXX)

# -DCMAKE_BUILD_TYPE=Debug|Release
SET(CMAKE_BUILD_TYPE Debug)
# SET(CMAKE_BUILD_TYPE Release)

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
    CMAKE_PREFIX_PATH
    ${CMAKE_PREFIX_PATH}
    deps/mathgl
    deps/opencv
    deps/libtorch
)

set(
    CMAKE_INSTALL_PREFIX
    ${CMAKE_SOURCE_DIR}/target/
)

set(
    CMAKE_INSTALL_INCLUDEDIR
    ${CMAKE_INSTALL_PREFIX}/include/
)

link_directories(
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    deps/mathgl/lib
    deps/opencv/x64/vc15/lib
    deps/libtorch/lib
)

include_directories(
    deps/mathgl/include
    deps/opencv/include
    deps/libtorch/include
    deps/libtorch/include/torch/csrc/api/include
)

find_package(Torch   REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(MathGL2 REQUIRED)

add_subdirectory(deps)
add_subdirectory(gg)
add_subdirectory(hh)
add_subdirectory(xx)
# add_subdirectory(yy)
