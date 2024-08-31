add_executable(testRAG TestRAG.cpp)
add_test(NAME testRAG COMMAND testRAG)
set_target_properties(testRAG PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testRAG
    lifurennlp
)
