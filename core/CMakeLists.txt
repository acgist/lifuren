project(lifurencore VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/Torch.cpp
    src/OpenCV.cpp
    src/audio/Audio.cpp
    src/audio/AudioModel.cpp
    src/image/Image.cpp
    src/image/ImageModel.cpp
    src/dataset/Dataset.cpp
    src/dataset/AudioDataset.cpp
    src/dataset/ImageDataset.cpp
)

if(CMAKE_HOST_WIN32)
    target_link_libraries(
        ${PROJECT_NAME}
        lifurenboot
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        ${FFMPEG_LIBRARIES}
    )
    target_include_directories(${PROJECT_NAME} PRIVATE ${FFMPEG_INCLUDE_DIRS})
else()
    target_link_libraries(
        ${PROJECT_NAME}
        lifurenboot
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        avutil
        avcodec
        avformat
        swresample
    )
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "module")

target_include_directories(${PROJECT_NAME} PUBLIC ${HEADER_DIR})

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
