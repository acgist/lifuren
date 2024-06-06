add_executable(testModel TestModel.cpp)
add_test(NAME testModel COMMAND testModel)
target_link_libraries(
    testModel PUBLIC
    lifurencore
)

add_executable(testTensor TestTensor.cpp)
add_test(NAME testTensor COMMAND testTensor)
target_link_libraries(
    testTensor PUBLIC
    lifurencore
)
