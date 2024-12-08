project(lifurencv VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/OpenCV.cpp
    src/audio/Audio.cpp
    src/audio/AudioDataset.cpp
    src/audio/ComposeClient.cpp
    src/image/Image.cpp
    src/image/PaintClient.cpp
    src/video/Video.cpp
    src/video/ActClient.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "module")

if(CMAKE_HOST_WIN32)
    target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${FFMPEG_LIBRARIES}
        lifurenmodel
        lifurenclient
    )
    target_include_directories(${PROJECT_NAME} PRIVATE ${FFMPEG_INCLUDE_DIRS})
else()
    target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        avutil
        avcodec
        avformat
        swresample
        lifurenmodel
        lifurenclient
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
