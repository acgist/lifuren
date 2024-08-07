project(lifurenclient VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME} STATIC
    src/source/Client.cpp
    src/source/RestClient.cpp
    src/source/ChatClient.cpp
    src/source/RestChatClient.cpp
    src/source/LocalChatClient.cpp
    src/source/EmbeddingClient.cpp
    src/source/RestEmbeddingClient.cpp
    src/source/LocalEmbeddingClient.cpp
    src/source/RAGClient.cpp
    src/source/PaintClient.cpp
    src/source/RestPaintClient.cpp
    src/source/LocalPaintClient.cpp
    src/source/VideoClient.cpp
    src/source/RestVideoClient.cpp
    src/source/LocalVideoClient.cpp
)

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    lifurencore
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
