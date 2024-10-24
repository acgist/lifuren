add_executable(testRAG TestRAG.cpp)
add_test(NAME testRAG COMMAND testRAG)
set_target_properties(testRAG PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testRAG
    lifurennlp
)

add_executable(testPoetry TestPoetry.cpp)
add_test(NAME testPoetry COMMAND testPoetry)
set_target_properties(testPoetry PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testPoetry
    lifurennlp
)

add_executable(testNLPModel TestNLPModel.cpp)
add_test(NAME testNLPModel COMMAND testNLPModel)
set_target_properties(testNLPModel PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testNLPModel
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
