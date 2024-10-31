option(JSON_BuildTests "是否编译JSON测试" OFF)

if(NOT TARGET json)
  add_subdirectory(json)
endif()

# 忽略所有依赖警告
# if(CMAKE_HOST_UNIX)
#   set_target_properties(
#     ...
#     PROPERTIES
#     FOLDER "deps"
#     COMPILE_FLAGS "-w"
#   )
# else()
#   set_target_properties(
#     ...
#     PROPERTIES
#     FOLDER "deps"
#     COMPILE_FLAGS "/wd4005"
#   )
# endif()
