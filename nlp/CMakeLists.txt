project(lifurennlp VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/RAGClient.cpp
    src/EmbeddingClient.cpp
    src/rag/FaissRAGClient.cpp
    src/rag/ElasticSearchRAGClient.cpp
    src/poetry/Poetry.cpp
    src/poetry/LiduModel.cpp
    src/poetry/SuxinModel.cpp
    src/poetry/PoetryDataset.cpp
    src/poetry/PoetizeClient.cpp
    src/embedding/OllamaEmbeddingClient.cpp
    src/embedding/PepperEmbeddingClient.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "module")

target_link_libraries(
    ${PROJECT_NAME}
    faiss
    lifurenmodel
    lifurenclient
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
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
