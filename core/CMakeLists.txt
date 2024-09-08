project(lifurencore VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/source/Dates.cpp
    src/source/Files.cpp
    src/source/Jsons.cpp
    src/source/Yamls.cpp
    src/source/Logger.cpp
    src/source/Strings.cpp
    src/source/Lifuren.cpp
    src/source/Exception.cpp
    src/source/config/Config.cpp
    src/source/config/Label.cpp
    src/source/config/LabelFile.cpp
    src/source/config/LabelText.cpp
    src/source/config/Mark.cpp
    src/source/config/MarkFile.cpp
    src/source/config/MarkText.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "module")

if(CMAKE_HOST_WIN32)
    target_link_libraries(
        ${PROJECT_NAME}
        spdlog::spdlog
        yaml-cpp::yaml-cpp
        nlohmann_json::nlohmann_json
    )
else()
    target_link_libraries(
        ${PROJECT_NAME}
        fmt
        spdlog
        yaml-cpp
        nlohmann_json::nlohmann_json
    )
endif()

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC ${HEADER_DIR}
)

file(COPY src/config/ DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/config)

# 开发环境配置
if(CMAKE_HOST_WIN32 AND (CMAKE_BUILD_TYPE MATCHES "Debug"))
    file(RENAME ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/config/config-dev.yml ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/config/config.yml)
endif()

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(DIRECTORY src/config/   DESTINATION ${CMAKE_INSTALL_PREFIX}/config)
