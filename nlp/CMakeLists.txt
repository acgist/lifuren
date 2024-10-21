project(lifurennlp VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/NLP.cpp
    src/Poetry.cpp
    src/RAGClient.cpp
    src/RAGService.cpp
    src/PoetryDataset.cpp
    src/PoetizeClient.cpp
    src/RAGTaskRunner.cpp
    src/EmbeddingClient.cpp
    src/rag/FaissRAGClient.cpp
    src/rag/ElasticSearchRAGClient.cpp
    src/poetize/ShifoPoetizeClient.cpp
    src/poetize/ShimoPoetizeClient.cpp
    src/poetize/ShiguiPoetizeClient.cpp
    src/poetize/ShixianPoetizeClient.cpp
    src/poetize/ShishengPoetizeClient.cpp
    src/poetize/LiduPoetizeClient.cpp
    src/poetize/SuxinPoetizeClient.cpp
    src/poetize/WanyuePoetizeClient.cpp
    src/embedding/OllamaEmbeddingClient.cpp
    src/embedding/ChineseWordVectorsEmbeddingClient.cpp
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
