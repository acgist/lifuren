project(lifurentest VERSION 1.0.0 LANGUAGES C CXX)

add_executable(${PROJECT_NAME} src/Main.cpp)
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "test")
target_link_libraries(
    ${PROJECT_NAME}
    lifurencore
)

include_directories(${HEADER_DIR})

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
    install(FILES ${OPENCV_DLLS}  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${SPDLOG_DLLS}  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${HTTPLIB_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FLTK_DLLS}    $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${YAML_DLLS}    $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FAISS_DLLS}   $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DLLS}  $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPDLOG_DLLS}  $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HTTPLIB_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()

add_subdirectory(src/cv)
add_subdirectory(src/nlp)
add_subdirectory(src/core)
add_subdirectory(src/boot)
add_subdirectory(src/model)
add_subdirectory(src/client)
