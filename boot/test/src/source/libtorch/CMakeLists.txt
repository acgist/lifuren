add_executable(testLibTorchYOLO TestLibTorchYOLO.cpp YOLO.cpp)
add_test(NAME testLibTorchYOLO COMMAND testLibTorchYOLO)
target_link_libraries(
    testLibTorchYOLO PUBLIC
    lifurencore
)

add_executable(testLibTorchModel TestLibTorchModel.cpp Model.cpp)
add_test(NAME testLibTorchModel COMMAND testLibTorchModel)
target_link_libraries(
    testLibTorchModel PUBLIC
    lifurencore
)

add_executable(testLibTorchTensor TestLibTorchTensor.cpp Tensor.cpp)
add_test(NAME testLibTorchTensor COMMAND testLibTorchTensor)
target_link_libraries(
    testLibTorchTensor PUBLIC
    lifurencore
)

add_executable(testLibTorchDeepLab TestLibTorchDeepLab.cpp DeepLab.cpp)
add_test(NAME testLibTorchDeepLab COMMAND testLibTorchDeepLab)
target_link_libraries(
    testLibTorchDeepLab PUBLIC
    lifurencore
)

add_executable(testLibTorchGenderHandler TestLibTorchGenderHandler.cpp GenderHandler.cpp)
add_test(NAME testLibTorchGenderHandler COMMAND testLibTorchGenderHandler)
target_link_libraries(
    testLibTorchGenderHandler PUBLIC
    lifurencore
)
