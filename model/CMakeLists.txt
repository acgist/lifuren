project(lifurenmodel VERSION 1.0.0 LANGUAGES C CXX)

option(ENABLE_TRAIN "打开训练模型" ON)

add_library(
    ${PROJECT_NAME} STATIC
    src/source/Model.cpp
    src/source/Models.cpp
    src/source/Datasets.cpp
    src/source/model/audio/Audio.cpp
    src/source/model/image/Image.cpp
    src/source/model/video/Video.cpp
    src/source/model/poetry/Poetry.cpp
    src/source/model/image/ImageToImageModel.cpp
    src/source/model/video/VideoToVideoModel.cpp
    src/source/model/image/LabelToImageModel.cpp
    src/source/model/image/PoetryToImageModel.cpp
    src/source/model/poetry/LabelToPoetryModel.cpp
    src/source/model/poetry/ImageToPoetryModel.cpp
)

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    lifurencore
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HEADER_DIR}
)

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
