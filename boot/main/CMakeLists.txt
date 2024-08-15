project(lifuren VERSION 1.0.0 LANGUAGES C CXX)

option(ENABLE_FLTK "开启FLTK界面" ON)
option(ENABLE_REST "开启REST接口" ON)

if(ENABLE_FLTK)
    message(STATUS "开启FLTK界面")
    add_definitions(-DLFR_ENABLE_FLTK=true)
    aux_source_directory(src/source/fltk          FLTK_FILES)
    aux_source_directory(src/source/fltk/app      FLTK_FILES)
    aux_source_directory(src/source/fltk/mark     FLTK_FILES)
    aux_source_directory(src/source/fltk/widget   FLTK_FILES)
    aux_source_directory(src/source/fltk/optimize FLTK_FILES)
endif()
    
if(ENABLE_REST)
    message(STATUS "开启REST接口")
    add_definitions(-DLFR_ENABLE_REST=true)
    aux_source_directory(src/source/rest     REST_FILES)
    aux_source_directory(src/source/rest/api REST_FILES)
endif()

add_executable(
    ${PROJECT_NAME}
    src/source/Main.cpp
    ${FLTK_FILES}
    ${REST_FILES}
)

if(CMAKE_HOST_WIN32)
    target_link_libraries(
        ${PROJECT_NAME}
        $<IF:$<TARGET_EXISTS:png>,png,png_shared>
        httplib::httplib
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
        cpp-httplib
        lifurencv
        lifurennlp
        ${FLTK_LIBRARIES}
    )
endif()

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC ${HEADER_DIR}
)

file(COPY src/images/ DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/images FILES_MATCHING PATTERN "*.png" PATTERN "*.jpg" PATTERN "*.jpeg")

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(DIRECTORY src/images/   DESTINATION ${CMAKE_INSTALL_PREFIX}/images)

if(CMAKE_HOST_WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRT")

    install(FILES ${FLTK_DLLS}   DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${YAML_DLLS}   DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${HTTP_DLLS}   DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${TORCH_DLLS}  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${OPENCV_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${PODOFO_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${SPDLOG_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FLTK_DLLS}   $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${YAML_DLLS}   $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HTTP_DLLS}   $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}  $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PODOFO_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPDLOG_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()
