if(NOT TARGET json)
  # set(JSON_BuildTests ON)
  add_subdirectory(json)
  # unset(JSON_BuildTests)
endif()

if(NOT USE_SD_GGML)
if(NOT TARGET ggml)
  # set(GGML_CUDA ON)
  add_subdirectory(ggml)
  # unset(GGML_CUDA)
endif()
endif()

if(NOT TARGET stable-diffusion.cpp)
  # set(SD_CUBLAS ON)
  add_subdirectory(stable-diffusion.cpp)
  # unset(SD_CUBLAS)
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