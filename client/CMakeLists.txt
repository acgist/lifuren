project(lifurenclient VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME} STATIC
    src/source/Client.cpp
    src/source/RestClient.cpp
    src/source/chat/ChatClient.cpp
    src/source/chat/OllamaChatClient.cpp
    src/source/chat/OpenAIChatClient.cpp
    src/source/paint/PaintClient.cpp
    src/source/paint/StableDiffusionCPPPaintClient.cpp
    src/source/embedding/EmbeddingClient.cpp
    src/source/embedding/CWVEmbeddingClient.cpp
    src/source/embedding/OllamaEmbeddingClient.cpp
)

target_link_libraries(
    ${PROJECT_NAME}
    lifurencore
)

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
