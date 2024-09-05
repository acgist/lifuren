add_executable(testFaiss TestFaiss.cpp)
add_test(NAME testFaiss COMMAND testFaiss)
set_target_properties(testFaiss PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testFaiss
    lifurennlp
)

add_executable(testRAGClient TestRAGClient.cpp)
add_test(NAME testRAGClient COMMAND testRAGClient)
set_target_properties(testRAGClient PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testRAGClient
    lifurennlp
)

add_executable(testChineseWordVectorsEmbeddingClient TestChineseWordVectorsEmbeddingClient.cpp)
add_test(NAME testChineseWordVectorsEmbeddingClient COMMAND testRAGClient)
set_target_properties(testChineseWordVectorsEmbeddingClient PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testChineseWordVectorsEmbeddingClient
    lifurennlp
)
