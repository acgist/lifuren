add_executable(testRAG TestRAG.cpp)
add_test(NAME testRAG COMMAND testRAG)
set_target_properties(testRAG PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testRAG
    lifurennlp
)

add_executable(testPepper TestPepper.cpp)
add_test(NAME testPepper COMMAND testPepper)
set_target_properties(testPepper PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testPepper
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

add_executable(testRAGClient TestRAGClient.cpp)
add_test(NAME testRAGClient COMMAND testRAGClient)
set_target_properties(testRAGClient PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testRAGClient
    lifurennlp
)

add_executable(testEmbeddingClient TestEmbeddingClient.cpp)
add_test(NAME testEmbeddingClient COMMAND testEmbeddingClient)
set_target_properties(testEmbeddingClient PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testEmbeddingClient
    lifurennlp
)
