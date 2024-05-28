add_executable(testYOLO TestYOLO.cpp)
add_test(NAME testYOLO COMMAND testYOLO)
target_link_libraries(
    testYOLO PUBLIC
    lifurencore
)

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

add_executable(testDeepLab TestDeepLab.cpp)
add_test(NAME testDeepLab COMMAND testDeepLab)
target_link_libraries(
    testDeepLab PUBLIC
    lifurencore
)

add_executable(testGenderHandler TestGenderHandler.cpp)
add_test(NAME testGenderHandler COMMAND testGenderHandler)
target_link_libraries(
    testGenderHandler PUBLIC
    lifurencore
)
