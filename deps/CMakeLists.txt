set(JSON_BuildTests OFF)
add_subdirectory(json)
unset(JSON_BuildTests)

# set(SD_CUBLAS ON)
add_subdirectory(stable-diffusion.cpp)
# unset(SD_CUBLAS)
