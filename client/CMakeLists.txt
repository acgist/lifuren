add_subdirectory(main)

if(ENABLE_TEST)
    enable_testing()
    message(STATUS "编译测试")
    add_subdirectory(test)
else()
    message(STATUS "忽略测试")
endif()

install(DIRECTORY webview/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/webview)

file(COPY webview/ DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin/webview)
