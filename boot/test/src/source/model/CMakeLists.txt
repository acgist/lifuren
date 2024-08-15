add_executable(testLayers TestLayers.cpp)
add_test(NAME testLayers COMMAND testLayers)
target_link_libraries(
    testLayers
    lifurenmodel
)

add_executable(testDatasets TestDatasets.cpp)
add_test(NAME testDatasets COMMAND testDatasets)
target_link_libraries(
    testDatasets
    lifurenmodel
)
