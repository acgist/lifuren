project(lifuren VERSION 1.0.0 LANGUAGES C CXX)

aux_source_directory(src/cli/ CLI_FILES)

if(ENABLE_FLTK)
    message(STATUS "编译FLTK界面")
    add_definitions(-DLFR_ENABLE_FLTK=true)
    aux_source_directory(src/fltk/ FLTK_FILES)
endif()
    
if(ENABLE_REST)
    message(STATUS "编译REST接口")
    add_definitions(-DLFR_ENABLE_REST=true)
    aux_source_directory(src/rest/ REST_FILES)
endif()

if(CMAKE_HOST_WIN32)
    add_executable(
        ${PROJECT_NAME}
        src/Main.cpp
        ${CLI_FILES}
        ${FLTK_FILES}
        ${REST_FILES}
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
        target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:WINDOWS")
    endif()
else()
    add_executable(
        ${PROJECT_NAME}
        src/Main.cpp
        ${CLI_FILES}
        ${FLTK_FILES}
        ${REST_FILES}
    )
endif()

if(CMAKE_HOST_WIN32)
    target_link_libraries(
        ${PROJECT_NAME}
        $<IF:$<TARGET_EXISTS:png>,png,png_shared>
        libjpeg-turbo::jpeg
        lifurencv
        lifurennlp
        ${FLTK_LIBRARIES}
    )
else()
    target_link_libraries(
        ${PROJECT_NAME}
        png
        jpeg
        lifurencv
        lifurennlp
        ${FLTK_LIBRARIES}
    )
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${HEADER_DIR})

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

if(CMAKE_HOST_WIN32)
    # /MD  MSVCRT.LIB   多线程DLL的Release版本
    # /MDd MSVCRTD.LIB  多线程DLL的Debug版本
    # /MT  LIBCMT.LIB   多线程静态链接的Release版本
    # /MTd LIBCMTD.LIB  多线程静态链接的Debug版本
    # /MD  MSVCPRT.LIB  多线程DLL的Release版本
    # /MDd MSVCPRTD.LIB 多线程DLL的Debug版本
    # /MT  LIBCPMT.LIB  多线程静态链接的Release版本
    # /MTd LIBCPMTD.LIB 多线程静态链接的Debug版本
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRT")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRTD")
    endif()

    install(FILES ${FLTK_DLLS}    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${YAML_DLLS}    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${FAISS_DLLS}   DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${TORCH_DLLS}   DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${FFMPEG_DLLS}  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${OPENCV_DLLS}  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${SPDLOG_DLLS}  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${HTTPLIB_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FLTK_DLLS}    $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${YAML_DLLS}    $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FAISS_DLLS}   $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}   $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DLLS}  $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DLLS}  $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPDLOG_DLLS}  $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HTTPLIB_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()
