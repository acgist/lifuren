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
    # 工具
    ${HEADER_DIR}/Ptr.hpp
    ${HEADER_DIR}/Dates.hpp
    ${HEADER_DIR}/Files.hpp
    ${HEADER_DIR}/Utils.hpp
    ${HEADER_DIR}/Logger.hpp
    ${HEADER_DIR}/Strings.hpp
    # 配置
    ${HEADER_DIR}/Mark.hpp
    ${HEADER_DIR}/Label.hpp
    ${HEADER_DIR}/Setting.hpp
    # 模型
    ${HEADER_DIR}/Model.hpp
    ${HEADER_DIR}/Audio.hpp
    ${HEADER_DIR}/AudioGC.hpp
    ${HEADER_DIR}/AudioTS.hpp
    ${HEADER_DIR}/Image.hpp
    ${HEADER_DIR}/ImageGC.hpp
    ${HEADER_DIR}/ImageTS.hpp
    ${HEADER_DIR}/Video.hpp
    ${HEADER_DIR}/VideoGC.hpp
    ${HEADER_DIR}/VideoTS.hpp
    ${HEADER_DIR}/Poetry.hpp
    ${HEADER_DIR}/PoetryGC.hpp
    ${HEADER_DIR}/PoetryTS.hpp
)

install(FILES ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME})

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)

install(DIRECTORY "src/config/" DESTINATION ${CMAKE_INSTALL_PREFIX}/config/)
