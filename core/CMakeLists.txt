project(lifurencore VERSION 1.0.0 LANGUAGES C CXX)

option(ENABLE_TRAIN "打开训练模型" ON)

add_library(
    ${PROJECT_NAME} STATIC
    src/source/Model.cpp
    src/source/Logger.cpp
    src/source/Lifuren.cpp
    src/source/config/Config.cpp
    src/source/config/Label.cpp
    src/source/config/LabelFile.cpp
    src/source/config/LabelText.cpp
    src/source/config/Mark.cpp
    src/source/config/MarkFile.cpp
    src/source/config/MarkText.cpp
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
    src/source/utils/Dates.cpp
    src/source/utils/Files.cpp
    src/source/utils/Jsons.cpp
    src/source/utils/Yamls.cpp
    src/source/utils/Layers.cpp
    src/source/utils/Models.cpp
    src/source/utils/Strings.cpp
    src/source/utils/Datasets.cpp
    src/source/utils/Collections.cpp
)

if(CMAKE_HOST_WIN32)
    target_link_libraries(
        ${PROJECT_NAME} PUBLIC
        spdlog::spdlog
        yaml-cpp::yaml-cpp
        nlohmann_json::nlohmann_json
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
    )
else()
    target_link_libraries(
        ${PROJECT_NAME} PUBLIC
        fmt
        spdlog
        yaml-cpp
        nlohmann_json::nlohmann_json
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
    )
endif()

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HEADER_DIR}
)

file(COPY src/config/ DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE_LOWER}/config)

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(DIRECTORY src/config/   DESTINATION ${CMAKE_INSTALL_PREFIX}/config)
