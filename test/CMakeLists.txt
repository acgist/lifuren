include_directories(${HEADER_DIR})

add_executable(testDataset src/TestDataset.cpp)
add_test(NAME testDataset COMMAND testDataset)
set_target_properties(testDataset PROPERTIES FOLDER "test")
target_link_libraries(
    testDataset
    lifurencore
)

add_executable(testWudaozi src/TestWudaozi.cpp)
add_test(NAME testWudaozi COMMAND testWudaozi)
set_target_properties(testWudaozi PROPERTIES FOLDER "test")
target_link_libraries(
    testWudaozi
    lifurencore
)

add_executable(testClassify src/TestClassify.cpp)
add_test(NAME testClassify COMMAND testClassify)
set_target_properties(testClassify PROPERTIES FOLDER "test")
target_link_libraries(
    testClassify
    lifurencore
)

add_executable(testLibtorch src/TestLibtorch.cpp)
add_test(NAME testLibtorch COMMAND testLibtorch)
set_target_properties(testLibtorch PROPERTIES FOLDER "test")
target_link_libraries(
    testLibtorch
    lifurencore
)
