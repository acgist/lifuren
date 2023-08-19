project(lifurenml VERSION 1.0.0 LANGUAGES C CXX)

aux_source_directory(${SOURCE_DIR} SOURCE_FILES)

add_executable(${PROJECT_NAME} src/Main.cpp ${SOURCE_FILES})

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    lifurengg
    mgl
    dlib::dlib
    ${OpenCV_LIBS}
    ${FLTK_LIBRARIES}
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HEADER_DIR}
)

set(
    HEADER_FILES
    ${HEADER_DIR}/DLibAll.hpp
    ${HEADER_DIR}/LifurenGG.hpp
    ${HEADER_DIR}/OpenCV.hpp
    ${HEADER_DIR}/Visual.hpp
)

install(
    FILES
    ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)

if(CMAKE_HOST_WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DLIB_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${FLTK_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${MATHGL_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENCV_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()
