# set(JSON_BuildTests ON)
add_subdirectory(json)
# unset(JSON_BuildTests)

# set(GGML_CUDA ON)
# add_subdirectory(ggml)
# unset(GGML_CUDA)

# set(SD_CUBLAS ON)
add_subdirectory(stable-diffusion.cpp)
# unset(SD_CUBLAS)

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