add_executable(testModel TestModel.cpp)
add_test(NAME testModel COMMAND testModel)
set_target_properties(testModel PROPERTIES FOLDER "test-core")
target_link_libraries(
    testModel
    lifurencore
)

add_executable(testLinear TestLinear.cpp)
add_test(NAME testLinear COMMAND testLinear)
set_target_properties(testLinear PROPERTIES FOLDER "test-core")
target_link_libraries(
    testLinear
    lifurencore
)

add_executable(testLibtorch TestLibtorch.cpp)
add_test(NAME testLibtorch COMMAND testLibtorch)
set_target_properties(testLibtorch PROPERTIES FOLDER "test-core")
target_link_libraries(
    testLibtorch
    lifurencore
)
