add_executable(testSuxin TestSuxin.cpp)
add_test(NAME testSuxin COMMAND testSuxin)
set_target_properties(testSuxin PROPERTIES FOLDER "test/nlp")
target_link_libraries(
    testSuxin
    lifurennlp
)

add_executable(testPoetry TestPoetry.cpp)
add_test(NAME testPoetry COMMAND testPoetry)
set_target_properties(testPoetry PROPERTIES FOLDER "test/nlp")
target_link_libraries(
    testPoetry
    lifurennlp
)

add_executable(testRAGClient TestRAGClient.cpp)
add_test(NAME testRAGClient COMMAND testRAGClient)
set_target_properties(testRAGClient PROPERTIES FOLDER "test/nlp")
target_link_libraries(
    testRAGClient
    lifurennlp
)

add_executable(testEmbeddingClient TestEmbeddingClient.cpp)
add_test(NAME testEmbeddingClient COMMAND testEmbeddingClient)
set_target_properties(testEmbeddingClient PROPERTIES FOLDER "test/nlp")
target_link_libraries(
    testEmbeddingClient
    lifurennlp
)
