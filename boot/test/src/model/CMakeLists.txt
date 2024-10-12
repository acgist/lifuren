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

add_executable(testLayerModel TestLayerModel.cpp)
add_test(NAME testLayerModel COMMAND testLayerModel)
set_target_properties(testLayerModel PROPERTIES FOLDER "test-model")
target_link_libraries(
    testLayerModel
    lifurenmodel
)
