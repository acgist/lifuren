add_executable(testLayer TestLayer.cpp)
add_test(NAME testLayer COMMAND testLayer)
set_target_properties(testLayer PROPERTIES FOLDER "test-model")
target_link_libraries(
    testLayer
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
    lifurenmodel
)

add_executable(testLinear TestLinear.cpp)
add_test(NAME testLinear COMMAND testLinear)
set_target_properties(testLinear PROPERTIES FOLDER "test-model")
target_link_libraries(
    testLinear
    lifurenmodel
)

add_executable(testTensor TestTensor.cpp)
add_test(NAME testTensor COMMAND testTensor)
set_target_properties(testTensor PROPERTIES FOLDER "test-model")
target_link_libraries(
    testTensor
    lifurenmodel
)

add_executable(testDataset TestDataset.cpp)
add_test(NAME testDataset COMMAND testDataset)
set_target_properties(testDataset PROPERTIES FOLDER "test-model")
target_link_libraries(
    testDataset
    lifurencv
    lifurennlp
    lifurenmodel
)
