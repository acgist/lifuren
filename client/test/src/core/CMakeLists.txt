add_executable(testDataset TestDataset.cpp)
add_test(NAME testDataset COMMAND testDataset)
set_target_properties(testDataset PROPERTIES FOLDER "test/core")
target_link_libraries(
    testDataset
    lifurencore
)

add_executable(testWudaozi TestWudaozi.cpp)
add_test(NAME testWudaozi COMMAND testWudaozi)
set_target_properties(testWudaozi PROPERTIES FOLDER "test/core")
target_link_libraries(
    testWudaozi
    lifurencore
)

add_executable(testLibtorch TestLibtorch.cpp)
add_test(NAME testLibtorch COMMAND testLibtorch)
set_target_properties(testLibtorch PROPERTIES FOLDER "test/core")
target_link_libraries(
    testLibtorch
    lifurencore
)

add_executable(testClassify TestClassify.cpp)
add_test(NAME testClassify COMMAND testClassify)
set_target_properties(testClassify PROPERTIES FOLDER "test/core")
target_link_libraries(
    testClassify
    lifurencore
)
