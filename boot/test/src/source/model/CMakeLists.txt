add_executable(testModel TestModel.cpp)
add_test(NAME testModel COMMAND testModel)
set_target_properties(testModel PROPERTIES FOLDER "test-model")
target_link_libraries(
    testModel
    lifurenmodel
)

add_executable(testLayers TestLayers.cpp)
add_test(NAME testLayers COMMAND testLayers)
set_target_properties(testLayers PROPERTIES FOLDER "test-model")
target_link_libraries(
    testLayers
    lifurenmodel
)

add_executable(testDatasets TestDatasets.cpp)
add_test(NAME testDatasets COMMAND testDatasets)
set_target_properties(testDatasets PROPERTIES FOLDER "test-model")
target_link_libraries(
    testDatasets
    lifurenmodel
)
