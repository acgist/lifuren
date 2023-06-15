cmake_minimum_required(VERSION 3.0.0)

project(lifurenaa VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -std=c17 -O0 -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -std=c17 -O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++17 -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17 -O3")


if(CMAKE_HOST_UNIX)
  include_directories()
elseif(CMAKE_HOST_WIN32)
  include_directories(D:/develop/libtorch/include)
  include_directories(D:/develop/libtorch/include/torch/csrc/api/include/torch)
else()
  message("不支持的系统")
endif()

set(
    HEADER_DIR
    src/header/
)

set(
    SOURCE_DIR
    src/source/
)

set(
    SOURCE_FILES
    src/Main.cpp
)

add_executable(
  ${PROJECT_NAME} ${SOURCE_FILES}
)
