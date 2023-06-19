cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example)
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} D:/develop/libtorch)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} D:/develop/libtorch D:/develop/opencv/build)
set(CMAKE_C_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
# set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")
# set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")
# set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
# set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
link_directories(
  D:/develop/libtorch/lib
  # D:/develop/opencv/build/x64/vc14/lib
)
include_directories(
  D:/develop/libtorch/include
  D:/develop/libtorch/include/torch/csrc/api/include
  # D:/develop/opencv/build/include
)
find_package(Torch REQUIRED)
# find_package(opencv REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(example example.cpp)
# target_link_libraries(example "${TORCH_LIBRARIES}")
message("1234", ${TORCH_LIBRARIES})
target_link_libraries(
    example PUBLIC
    -lc10
    -ltorch
    # -lkineto
    -ltorch_cpu
)