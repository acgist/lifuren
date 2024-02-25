add_subdirectory(main)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
message("Debug模式打开测试")
add_subdirectory(test)
else()
message("非Debug模式关闭测试")
endif()
