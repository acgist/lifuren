add_executable(testRAG TestRAG.cpp)
add_test(NAME testRAG COMMAND testRAG)
target_link_libraries(
    testRAG
    lifurennlp
)
