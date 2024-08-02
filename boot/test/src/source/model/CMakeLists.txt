add_executable(testLayers TestLayers.cpp)
add_test(NAME testLayers COMMAND testLayers)
target_link_libraries(
    testLayers PUBLIC
    lifurenmodel
)

add_executable(testDatasets TestDatasets.cpp)
add_test(NAME testDatasets COMMAND testDatasets)
target_link_libraries(
    testDatasets PUBLIC
    lifurenmodel
)
