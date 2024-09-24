add_executable(testRAG TestRAG.cpp)
add_test(NAME testRAG COMMAND testRAG)
set_target_properties(testRAG PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testRAG
    lifurennlp
)

add_executable(testFaiss TestFaiss.cpp)
add_test(NAME testFaiss COMMAND testFaiss)
set_target_properties(testFaiss PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testFaiss
    lifurennlp
)

add_executable(testPoetrys TestPoetrys.cpp)
add_test(NAME testPoetrys COMMAND testPoetrys)
set_target_properties(testPoetrys PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testPoetrys
    lifurennlp
)

add_executable(testFaissRAGClient TestFaissRAGClient.cpp)
add_test(NAME testFaissRAGClient COMMAND testFaissRAGClient)
set_target_properties(testFaissRAGClient PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testFaissRAGClient
    lifurennlp
)

add_executable(testOllamaEmbeddingClient TestOllamaEmbeddingClient.cpp)
add_test(NAME testOllamaEmbeddingClient COMMAND testOllamaEmbeddingClient)
set_target_properties(testOllamaEmbeddingClient PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testOllamaEmbeddingClient
    lifurennlp
)

add_executable(testElasticSearchRAGClient TestElasticSearchRAGClient.cpp)
add_test(NAME testElasticSearchRAGClient COMMAND testElasticSearchRAGClient)
set_target_properties(testElasticSearchRAGClient PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testElasticSearchRAGClient
    lifurennlp
)

add_executable(testChineseWordVectorsEmbeddingClient TestChineseWordVectorsEmbeddingClient.cpp)
add_test(NAME testChineseWordVectorsEmbeddingClient COMMAND testChineseWordVectorsEmbeddingClient)
set_target_properties(testChineseWordVectorsEmbeddingClient PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testChineseWordVectorsEmbeddingClient
    lifurennlp
)
