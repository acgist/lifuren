project(lifurencore VERSION 1.0.0 LANGUAGES C CXX)

aux_source_directory(${SOURCE_DIR} SOURCE_FILES)

file(GLOB SUB_SOURCE_FILES "src/source/*/*.cpp")

add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES} ${SUB_SOURCE_FILES})

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
    spdlog::spdlog
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

set(
    HEADER_FILES
    # 通用
    ${HEADER_DIR}/Ptr.hpp
    ${HEADER_DIR}/Ref.hpp
    ${HEADER_DIR}/Model.hpp
    ${HEADER_DIR}/Logger.hpp
    ${HEADER_DIR}/Lifuren.hpp
    # 工具
    ${HEADER_DIR}/utils/Dates.hpp
    ${HEADER_DIR}/utils/Files.hpp
    ${HEADER_DIR}/utils/Jsons.hpp
    ${HEADER_DIR}/utils/Yamls.hpp
    ${HEADER_DIR}/utils/Strings.hpp
    ${HEADER_DIR}/utils/Datasets.hpp
    ${HEADER_DIR}/utils/Collections.hpp
    # 配置
    ${HEADER_DIR}/config/Mark.hpp
    ${HEADER_DIR}/config/Label.hpp
    ${HEADER_DIR}/config/Config.hpp
    # 模型
    ${HEADER_DIR}/model/Audio.hpp
    ${HEADER_DIR}/model/AudioGC.hpp
    ${HEADER_DIR}/model/AudioTS.hpp
    ${HEADER_DIR}/model/Image.hpp
    ${HEADER_DIR}/model/ImageGC.hpp
    ${HEADER_DIR}/model/ImageTS.hpp
    ${HEADER_DIR}/model/Video.hpp
    ${HEADER_DIR}/model/VideoGC.hpp
    ${HEADER_DIR}/model/VideoTS.hpp
    ${HEADER_DIR}/model/Poetry.hpp
    ${HEADER_DIR}/model/PoetryGC.hpp
    ${HEADER_DIR}/model/PoetryTS.hpp
    # 模型工具
    ${HEADER_DIR}/handler/Handler.hpp
    # 微调
    ${HEADER_DIR}/finetune/Finetune.hpp
    # 奖励
    ${HEADER_DIR}/prompt/Tokenizer.hpp
    ${HEADER_DIR}/prompt/Embedding.hpp
)

install(FILES ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME})

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)

install(DIRECTORY src/config/ DESTINATION ${CMAKE_INSTALL_PREFIX}/config/)

file(COPY src/config/ DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE_LOWER}/config/)
