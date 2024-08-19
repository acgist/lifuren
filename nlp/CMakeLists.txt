project(lifurennlp VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME} STATIC
    src/source/NLP.cpp
    src/source/reader/DocumentReader.cpp
    src/source/reader/PDFReader.cpp
    src/source/reader/TextReader.cpp
    src/source/reader/WordReader.cpp
    src/source/reader/TextReader.cpp
    src/source/reader/MarkdownReader.cpp
    src/source/chunk/ChunkService.cpp
    src/source/chunk/ChunkStrategy.cpp
    src/source/chunk/LineChunkStrategy.cpp
    src/source/rag/RAGClient.cpp
    src/source/rag/RAGService.cpp
    src/source/rag/RAGTaskRunner.cpp
    src/source/rag/ElasticSearchRAGClient.cpp
)

target_link_libraries(
    ${PROJECT_NAME}
    minidocx
    $<IF:$<TARGET_EXISTS:podofo_shared>,podofo_shared,podofo_static>
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
