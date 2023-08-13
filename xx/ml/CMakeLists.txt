project(lifurenml VERSION 1.0.0 LANGUAGES C CXX)

set(HEADER_DIR src/header/)
set(SOURCE_DIR src/source/)

aux_source_directory(${SOURCE_DIR} SOURCE_FILES)

add_executable(${PROJECT_NAME} src/Main.cpp ${SOURCE_FILES})

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    lifurengg
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HEADER_DIR}
)

set(
    HEADER_FILES
    ${HEADER_DIR}/DLibAll.hpp
    ${HEADER_DIR}/LifurenGG.hpp
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
    file(GLOB MATHGL_DLLS "${MathGL2_LIBRARIES_DIRS}/../bin/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${MATHGL_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()

if(CMAKE_HOST_WIN32)
    file(GLOB OPENCV_DLLS "${OpenCV_INSTALL_PATH}/x64/vc15/bin/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENCV_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()
