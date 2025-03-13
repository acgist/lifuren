add_subdirectory(main)

if(ENABLE_TEST)
    enable_testing()
    message(STATUS "编译TEST代码")
    add_subdirectory(test)
else()
    message(STATUS "忽略TEST代码")
endif()
