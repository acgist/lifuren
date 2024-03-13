project(lifurencore VERSION 1.0.0 LANGUAGES C CXX)

aux_source_directory(${SOURCE_DIR} SOURCE_FILES)

file(GLOB SUB_SOURCE_FILES "src/source/*/*.cpp")

add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES} ${SUB_SOURCE_FILES})

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    ${TORCH_LIBRARIES}
)

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
    # 工具
    ${HEADER_DIR}/utils/Dates.hpp
    ${HEADER_DIR}/utils/Files.hpp
    ${HEADER_DIR}/utils/Jsons.hpp
    ${HEADER_DIR}/utils/Utils.hpp
    ${HEADER_DIR}/utils/Strings.hpp
    ${HEADER_DIR}/utils/Collections.hpp
    # 配置
    ${HEADER_DIR}/config/Mark.hpp
    ${HEADER_DIR}/config/Label.hpp
    ${HEADER_DIR}/config/Setting.hpp
    # 模型
    ${HEADER_DIR}/audio/Audio.hpp
    ${HEADER_DIR}/audio/AudioGC.hpp
    ${HEADER_DIR}/audio/AudioTS.hpp
    ${HEADER_DIR}/image/Image.hpp
    ${HEADER_DIR}/image/ImageGC.hpp
    ${HEADER_DIR}/image/ImageTS.hpp
    ${HEADER_DIR}/video/Video.hpp
    ${HEADER_DIR}/video/VideoGC.hpp
    ${HEADER_DIR}/video/VideoTS.hpp
    ${HEADER_DIR}/poetry/Poetry.hpp
    ${HEADER_DIR}/poetry/PoetryGC.hpp
    ${HEADER_DIR}/poetry/PoetryTS.hpp
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
