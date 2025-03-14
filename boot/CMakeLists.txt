project(lifurenboot VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/Date.cpp
    src/File.cpp
    src/Config.cpp
    src/Logger.cpp
    src/String.cpp
    src/Message.cpp
)

if(CMAKE_HOST_WIN32)
    target_link_libraries(
        ${PROJECT_NAME}
        spdlog::spdlog
        yaml-cpp::yaml-cpp
    )
else()
    target_link_libraries(
        ${PROJECT_NAME}
        fmt
        spdlog
        yaml-cpp
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
    
if(CMAKE_HOST_WIN32)
    set(CONFIG_FILES "config/config-win.yml")
else()
    set(CONFIG_FILES "config/config.yml")
endif()

install(FILES ${CONFIG_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/config)

file(COPY ${CONFIG_FILES} DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/config)
