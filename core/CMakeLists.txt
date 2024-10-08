project(lifurencore VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/Dates.cpp
    src/Files.cpp
    src/Yamls.cpp
    src/Config.cpp
    src/Logger.cpp
    src/Rhythm.cpp
    src/Lifuren.cpp
    src/Strings.cpp
    src/Exception.cpp
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

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})

file(COPY config/ DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/config)

install(DIRECTORY config/ DESTINATION ${CMAKE_INSTALL_PREFIX}/config)
