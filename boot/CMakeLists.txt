add_subdirectory(main)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
add_subdirectory(test)
else()
# 忽略测试代码
endif()
