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
