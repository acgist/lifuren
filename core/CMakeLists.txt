project(lifurencore VERSION 1.0.0 LANGUAGES C CXX)

aux_source_directory(${SOURCE_DIR} SOURCE_FILES)

file(GLOB SUB_SOURCE_FILES "src/source/*/*.cpp" "src/source/*/*/*.cpp")

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

file(COPY src/config/ DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE_LOWER}/config/)

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME})
install(DIRECTORY src/config/   DESTINATION ${CMAKE_INSTALL_PREFIX}/config/)
