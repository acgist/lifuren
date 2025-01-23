add_executable(testLidu TestLidu.cpp)
add_test(NAME testLidu COMMAND testLidu)
set_target_properties(testLidu PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testLidu
    lifurennlp
)
add_executable(testSuxin TestSuxin.cpp)
add_test(NAME testSuxin COMMAND testSuxin)
set_target_properties(testSuxin PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testSuxin
    lifurennlp
)

add_executable(testPoetry TestPoetry.cpp)
add_test(NAME testPoetry COMMAND testPoetry)
set_target_properties(testPoetry PROPERTIES FOLDER "test-nlp")
target_link_libraries(
    testPoetry
    lifurennlp
)
