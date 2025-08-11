project(lifurencore VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/File.cpp
    src/Config.cpp
    src/Logger.cpp
    src/Dataset.cpp
    src/Wudaozi.cpp
)

if(CMAKE_HOST_WIN32)
    target_link_libraries(
        ${PROJECT_NAME}
        spdlog::spdlog
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
    )
else()
    target_link_libraries(
        ${PROJECT_NAME}
        fmt
        spdlog
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
    )
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${HEADER_DIR})

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
