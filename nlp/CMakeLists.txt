project(lifurennlp VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/NLP.cpp
    src/Poetrys.cpp
    src/RAGClient.cpp
    src/RAGService.cpp
    src/PoetizeClient.cpp
    src/RAGTaskRunner.cpp
    src/PoetryDatasets.cpp
    src/EmbeddingClient.cpp
    src/rag/FaissRAGClient.cpp
    src/rag/ElasticSearchRAGClient.cpp
    src/model/ShifoRNNModel.cpp
    src/model/ShimoRNNModel.cpp
    src/model/ShiguiRNNModel.cpp
    src/model/ShixianRNNModel.cpp
    src/model/ShishengRNNModel.cpp
    src/model/LiduRNNModel.cpp
    src/model/SuxinRNNModel.cpp
    src/model/WanyueRNNModel.cpp
    src/poetize/ShifoRNNPoetizeClient.cpp
    src/poetize/ShimoRNNPoetizeClient.cpp
    src/poetize/ShiguiRNNPoetizeClient.cpp
    src/poetize/ShixianRNNPoetizeClient.cpp
    src/poetize/ShishengRNNPoetizeClient.cpp
    src/poetize/LiduRNNPoetizeClient.cpp
    src/poetize/SuxinRNNPoetizeClient.cpp
    src/poetize/WanyueRNNPoetizeClient.cpp
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
