project(lifuren VERSION 1.0.0 LANGUAGES C CXX)

aux_source_directory(${SOURCE_DIR} SOURCE_FILES)

file(GLOB SUB_SOURCE_FILES "src/source/*/*.cpp")

add_executable(${PROJECT_NAME} src/Main.cpp ${SOURCE_FILES} ${SUB_SOURCE_FILES})

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    png
    lifurencore
    ${FLTK_LIBRARIES}
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HEADER_DIR}
)

set(
    HEADER_FILES
    ${HEADER_DIR}/Window.hpp
)

install(FILES ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME})

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)

set(
    IMAGES_FILES
    src/images/logo.png
)

install(FILES ${IMAGES_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/images/)

if(CMAKE_HOST_WIN32)
    install(FILES ${FLTK_DLLS}  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${TORCH_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
endif()

if(CMAKE_HOST_WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRT")
    
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FLTK_DLLS}  $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
    
    foreach(image_file ${IMAGES_FILES})
        file(COPY ${image_file} DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE_LOWER}/images)
    endforeach(image_file)
endif()
