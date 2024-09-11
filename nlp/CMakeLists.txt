project(lifurennlp VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/source/NLP.cpp
    src/source/RAGClient.cpp
    src/source/RAGService.cpp
    src/source/RAGTaskRunner.cpp
    src/source/EmbeddingClient.cpp
    src/source/rag/FaissRAGClient.cpp
    src/source/rag/ElasticSearchRAGClient.cpp
    src/source/model/ShifoModel.cpp
    src/source/model/ShimoModel.cpp
    src/source/model/ShiguiModel.cpp
    src/source/model/ShixianModel.cpp
    src/source/model/ShishengModel.cpp
    src/source/model/LiduModel.cpp
    src/source/model/SuxinModel.cpp
    src/source/model/WanyueModel.cpp
    src/source/poetize/ShifoRNNPoetizeClient.cpp
    src/source/poetize/ShimoRNNPoetizeClient.cpp
    src/source/poetize/ShiguiRNNPoetizeClient.cpp
    src/source/poetize/ShixianRNNPoetizeClient.cpp
    src/source/poetize/ShishengRNNPoetizeClient.cpp
    src/source/poetize/LiduRNNPoetizeClient.cpp
    src/source/poetize/SuxinRNNPoetizeClient.cpp
    src/source/poetize/WanyueRNNPoetizeClient.cpp
    src/source/embedding/OllamaEmbeddingClient.cpp
    src/source/embedding/ChineseWordVectorsEmbeddingClient.cpp
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
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
