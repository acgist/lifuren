# set(JSON_BuildTests ON)
add_subdirectory(json)
# unset(JSON_BuildTests)

# set(GGML_CUDA ON)
# add_subdirectory(ggml)
# unset(GGML_CUDA)

# set(SD_CUBLAS ON)
add_subdirectory(stable-diffusion.cpp)
# unset(SD_CUBLAS)

set_target_properties(zip              PROPERTIES FOLDER "deps")
set_target_properties(ggml             PROPERTIES FOLDER "deps")
set_target_properties(stable-diffusion PROPERTIES FOLDER "deps")
