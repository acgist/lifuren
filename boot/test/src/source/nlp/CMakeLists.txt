add_executable(testRAG TestRAG.cpp)
add_test(NAME testRAG COMMAND testRAG)
target_link_libraries(
    testRAG
    lifurennlp
)

add_executable(testDocumentChunk TestDocumentChunk.cpp)
add_test(NAME testDocumentChunk COMMAND testDocumentChunk)
target_link_libraries(
    testDocumentChunk
    lifurennlp
)

add_executable(testDocumentReader TestDocumentReader.cpp)
add_test(NAME testDocumentReader COMMAND testDocumentReader)
target_link_libraries(
    testDocumentReader
    lifurennlp
)
