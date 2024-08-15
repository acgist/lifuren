add_executable(testChunkStrategy TestChunkStrategy.cpp)
add_test(NAME testChunkStrategy COMMAND testChunkStrategy)
target_link_libraries(
    testChunkStrategy
    lifurennlp
)

add_executable(testDocumentReader TestDocumentReader.cpp)
add_test(NAME testDocumentReader COMMAND testDocumentReader)
target_link_libraries(
    testDocumentReader
    lifurennlp
)
