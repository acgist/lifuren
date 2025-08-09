project(lifuren VERSION 1.0.0 LANGUAGES C CXX)

# include(CPack)

aux_source_directory(src/ CLI_FILES)

if(CMAKE_HOST_WIN32)
    add_executable(
        ${PROJECT_NAME}
        ${CLI_FILES}
        resources/win.rc
    )
    # /SUBSYSTEM:CONSOLE
    # /SUBSYSTEM:WINDOWS
    # add_link_options(${PROJECT_NAME} PRIVATE -mconsole)
    # add_link_options(${PROJECT_NAME} PRIVATE -mwindows)
    # target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:CONSOLE")
    # target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:WINDOWS")
    # set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_DEBUG   "/SUBSYSTEM:CONSOLE")
    # set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:WINDOWS")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:CONSOLE")
    else()
        target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:CONSOLE")
    endif()
else()
    add_executable(
        ${PROJECT_NAME}
        ${CLI_FILES}
    )
endif()

target_link_libraries(
    ${PROJECT_NAME}
    lifurencore
)

target_include_directories(${PROJECT_NAME} PUBLIC ${HEADER_DIR})

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

if(CMAKE_HOST_WIN32)
    # C = C | CP = C++
    # /MD  MSVCRT.LIB   多线程DLL的Release版本
    # /MDd MSVCRTD.LIB  多线程DLL的Debug版本
    # /MD  MSVCPRT.LIB  多线程DLL的Release版本
    # /MDd MSVCPRTD.LIB 多线程DLL的Debug版本
    # /MT  LIBCMT.LIB   多线程静态链接的Release版本
    # /MTd LIBCMTD.LIB  多线程静态链接的Debug版本
    # /MT  LIBCPMT.LIB  多线程静态链接的Release版本
    # /MTd LIBCPMTD.LIB 多线程静态链接的Debug版本
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRT")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRTD /entry:mainCRTStartup")
    endif()

    install(FILES ${TORCH_DLLS}  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${OPENCV_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${SPDLOG_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}  $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPDLOG_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()
