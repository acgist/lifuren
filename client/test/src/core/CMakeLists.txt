add_executable(testAudio TestAudio.cpp)
add_test(NAME testAudio COMMAND testAudio)
set_target_properties(testAudio PROPERTIES FOLDER "test-core")
target_link_libraries(
    testAudio
    lifurencore
)

add_executable(testImage TestImage.cpp)
add_test(NAME testImage COMMAND testImage)
set_target_properties(testImage PROPERTIES FOLDER "test-core")
target_link_libraries(
    testImage
    lifurencore
)

add_executable(testVideo TestVideo.cpp)
add_test(NAME testVideo COMMAND testVideo)
set_target_properties(testVideo PROPERTIES FOLDER "test-core")
target_link_libraries(
    testVideo
    lifurencore
)

add_executable(testModel TestModel.cpp)
add_test(NAME testModel COMMAND testModel)
set_target_properties(testModel PROPERTIES FOLDER "test-core")
target_link_libraries(
    testModel
    lifurencore
)

add_executable(testGender TestGender.cpp)
add_test(NAME testGender COMMAND testGender)
set_target_properties(testGender PROPERTIES FOLDER "test-core")
target_link_libraries(
    testGender
    lifurencore
)

add_executable(testWudaozi TestWudaozi.cpp)
add_test(NAME testWudaozi COMMAND testWudaozi)
set_target_properties(testWudaozi PROPERTIES FOLDER "test-core")
target_link_libraries(
    testWudaozi
    lifurencore
)

add_executable(testShikuang TestShikuang.cpp)
add_test(NAME testShikuang COMMAND testShikuang)
set_target_properties(testShikuang PROPERTIES FOLDER "test-core")
target_link_libraries(
    testShikuang
    lifurencore
)

add_executable(testLibtorch TestLibtorch.cpp)
add_test(NAME testLibtorch COMMAND testLibtorch)
set_target_properties(testLibtorch PROPERTIES FOLDER "test-core")
target_link_libraries(
    testLibtorch
    lifurencore
)
