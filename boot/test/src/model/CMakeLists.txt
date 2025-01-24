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

add_executable(testLibtorch TestLibtorch.cpp)
add_test(NAME testLibtorch COMMAND testLibtorch)
set_target_properties(testLibtorch PROPERTIES FOLDER "test-model")
target_link_libraries(
    testLibtorch
    lifurenmodel
)
