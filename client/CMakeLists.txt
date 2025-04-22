add_subdirectory(main)

if(ENABLE_TEST)
    enable_testing()
    message(STATUS "编译测试")
    add_subdirectory(test)
else()
    message(STATUS "忽略测试")
endif()
