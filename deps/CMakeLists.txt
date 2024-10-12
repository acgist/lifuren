option(SD_CUBLAS       "是否编译CUDA依赖" OFF)
option(GGML_CUDA       "是否编译CUDA依赖" OFF)
option(JSON_BuildTests "是否编译JSON测试" OFF)

if(NOT TARGET json)
  add_subdirectory(json)
endif()

if(NOT USE_SD_GGML)
if(NOT TARGET ggml)
  add_subdirectory(ggml)
endif()
endif()

if(NOT TARGET stable-diffusion.cpp)
  add_subdirectory(stable-diffusion.cpp)
endif()

# 忽略所有依赖警告
if(CMAKE_HOST_UNIX)
  set_target_properties(
    zip
    ggml
    stable-diffusion
    PROPERTIES
    FOLDER "deps"
    COMPILE_FLAGS "-w"
  )
else()
  set_target_properties(
    zip
    ggml
    stable-diffusion
    PROPERTIES
    FOLDER "deps"
    COMPILE_FLAGS "/wd4005"
  )
endif()