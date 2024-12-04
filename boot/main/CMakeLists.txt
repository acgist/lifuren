project(lifuren VERSION 1.0.0 LANGUAGES C CXX)

aux_source_directory(src/cli/ CLI_FILES)

if(ENABLE_FLTK)
    message(STATUS "开启FLTK界面")
    add_definitions(-DLFR_ENABLE_FLTK=true)
    aux_source_directory(src/fltk/ FLTK_FILES)
endif()
    
if(ENABLE_REST)
    message(STATUS "开启REST接口")
    add_definitions(-DLFR_ENABLE_REST=true)
    aux_source_directory(src/rest/     REST_FILES)
    aux_source_directory(src/rest/api/ REST_FILES)
endif()

add_executable(
    ${PROJECT_NAME}
    src/Main.cpp
    ${CLI_FILES}
    ${FLTK_FILES}
    ${REST_FILES}
)

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

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC ${HEADER_DIR}
)

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

if(CMAKE_HOST_WIN32)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRT")
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
