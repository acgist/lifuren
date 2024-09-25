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

add_executable(testEmbeddingClient TestEmbeddingClient.cpp)
add_test(NAME testEmbeddingClient COMMAND testEmbeddingClient)
set_target_properties(testEmbeddingClient PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testEmbeddingClient
    lifurennlp
)

add_executable(testElasticSearchRAGClient TestElasticSearchRAGClient.cpp)
add_test(NAME testElasticSearchRAGClient COMMAND testElasticSearchRAGClient)
set_target_properties(testElasticSearchRAGClient PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testElasticSearchRAGClient
    lifurennlp
)
