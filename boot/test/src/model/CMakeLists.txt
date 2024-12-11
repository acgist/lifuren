add_executable(testRNN TestRNN.cpp)
add_test(NAME testRNN COMMAND testRNN)
set_target_properties(testRNN PROPERTIES FOLDER "test-model")
target_link_libraries(
    testRNN
    lifurenmodel
)

add_executable(testJit TestJit.cpp)
add_test(NAME testJit COMMAND testJit)
set_target_properties(testJit PROPERTIES FOLDER "test-model")
target_link_libraries(
    testJit
    lifurenmodel
)

add_executable(testOnnx TestOnnx.cpp)
add_test(NAME testOnnx COMMAND testOnnx)
set_target_properties(testOnnx PROPERTIES FOLDER "test-model")
target_link_libraries(
    testOnnx
    lifurenmodel
)

add_executable(testModel TestModel.cpp)
add_test(NAME testModel COMMAND testModel)
set_target_properties(testModel PROPERTIES FOLDER "test-model")
target_link_libraries(
    testModel
    lifurenmodel
)

add_executable(testGender TestGender.cpp)
add_test(NAME testGender COMMAND testGender)
set_target_properties(testGender PROPERTIES FOLDER "test-model")
target_link_libraries(
    testGender
    lifurencv
)

add_executable(testLinear TestLinear.cpp)
add_test(NAME testLinear COMMAND testLinear)
set_target_properties(testLinear PROPERTIES FOLDER "test-model")
target_link_libraries(
    testLinear
    lifurenmodel
)

add_executable(testDataset TestDataset.cpp)
add_test(NAME testDataset COMMAND testDataset)
set_target_properties(testDataset PROPERTIES FOLDER "test-model")
target_link_libraries(
    testDataset
    lifurencv
    lifurennlp
)

add_executable(testLibtorch TestLibtorch.cpp)
add_test(NAME testLibtorch COMMAND testLibtorch)
set_target_properties(testLibtorch PROPERTIES FOLDER "test-model")
target_link_libraries(
    testLibtorch
    lifurenmodel
)
