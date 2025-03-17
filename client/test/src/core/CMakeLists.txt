add_executable(testGender TestGender.cpp)
add_test(NAME testGender COMMAND testGender)
set_target_properties(testGender PROPERTIES FOLDER "test-core")
target_link_libraries(
    testGender
    lifurencore
)

add_executable(testLinear TestLinear.cpp)
add_test(NAME testLinear COMMAND testLinear)
set_target_properties(testLinear PROPERTIES FOLDER "test-core")
target_link_libraries(
    testLinear
    lifurencore
)

add_executable(testDataset TestDataset.cpp)
add_test(NAME testDataset COMMAND testDataset)
set_target_properties(testDataset PROPERTIES FOLDER "test-core")
target_link_libraries(
    testDataset
    lifurencore
)

add_executable(testLibtorch TestLibtorch.cpp)
add_test(NAME testLibtorch COMMAND testLibtorch)
set_target_properties(testLibtorch PROPERTIES FOLDER "test-core")
target_link_libraries(
    testLibtorch
    lifurencore
)
