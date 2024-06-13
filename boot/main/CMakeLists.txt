project(lifuren VERSION 1.0.0 LANGUAGES C CXX)

add_executable(
    ${PROJECT_NAME}
    src/source/Main.cpp
    src/source/fltk/FLTK.cpp
    src/source/fltk/MainWindow.cpp
    src/source/fltk/AboutWindow.cpp
    src/source/fltk/AudioGCWindow.cpp
    src/source/fltk/AudioTSWindow.cpp
    src/source/fltk/ImageGCWindow.cpp
    src/source/fltk/ImageTSWindow.cpp
    src/source/fltk/PoetryGCWindow.cpp
    src/source/fltk/PoetryTSWindow.cpp
    src/source/fltk/VideoGCWindow.cpp
    src/source/fltk/VideoTSWindow.cpp
    src/source/rest/REST.cpp
    src/source/rest/AudioAPI.cpp
    src/source/rest/ImageAPI.cpp
    src/source/rest/PoetryAPI.cpp
    src/source/rest/VideoAPI.cpp
)

add_definitions(-D__FLTK__=true)
add_definitions(-D__REST__=true)

if(CMAKE_HOST_WIN32)
if(TARGET png)
target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    png
    httplib::httplib
    libjpeg-turbo::jpeg
    lifurencore
    ${FLTK_LIBRARIES}
)
else()
target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    png_shared
    httplib::httplib
    libjpeg-turbo::jpeg
    lifurencore
    ${FLTK_LIBRARIES}
)
endif()
else()
target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    png
    jpeg
    cpp-httplib
    lifurencore
    ${FLTK_LIBRARIES}
)
endif()

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HEADER_DIR}
)

file(COPY src/images/ DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE_LOWER}/images/ FILES_MATCHING PATTERN "*.png" PATTERN "*.jpg" PATTERN "*.jpeg")

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME})
install(DIRECTORY src/images/   DESTINATION ${CMAKE_INSTALL_PREFIX}/images/)

if(CMAKE_HOST_WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRT")

    install(FILES ${FLTK_DLLS}   DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${YAML_DLLS}   DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${HTTP_DLLS}   DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${TORCH_DLLS}  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${OPENCV_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${SPDLOG_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FLTK_DLLS}   $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${YAML_DLLS}   $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HTTP_DLLS}   $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}  $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPDLOG_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()
