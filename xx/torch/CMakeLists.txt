cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(my_test)
set(CMAKE_PREFIX_PATH D:/develop/libtorch)
link_directories(
  D:/develop/libtorch/lib
)
include_directories(
  D:/develop/libtorch/include
  D:/develop/libtorch/include/torch/csrc/api/include
)
find_package(Torch REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}")
set_property(TARGET example PROPERTY CXX_STANDARD 17)
# target_link_libraries(
#     ${PROJECT_NAME} PUBLIC
#     -lc10
#     -ltorch
#     -ltorch_cpu
# )