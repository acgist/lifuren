add_executable(testAudio TestAudio.cpp)
add_test(NAME testAudio COMMAND testAudio)
set_target_properties(testAudio PROPERTIES FOLDER "test-cv")
target_link_libraries(
    testAudio
    lifurencv
)

add_executable(testImage TestImage.cpp)
add_test(NAME testImage COMMAND testImage)
set_target_properties(testImage PROPERTIES FOLDER "test-cv")
target_link_libraries(
    testImage
    lifurencv
)

add_executable(testVideo TestVideo.cpp)
add_test(NAME testVideo COMMAND testVideo)
set_target_properties(testVideo PROPERTIES FOLDER "test-cv")
target_link_libraries(
    testVideo
    lifurencv
)

add_executable(testWudaozi TestWudaozi.cpp)
add_test(NAME testWudaozi COMMAND testWudaozi)
set_target_properties(testWudaozi PROPERTIES FOLDER "test-cv")
target_link_libraries(
    testWudaozi
    lifurencv
)

add_executable(testShikuang TestShikuang.cpp)
add_test(NAME testShikuang COMMAND testShikuang)
set_target_properties(testShikuang PROPERTIES FOLDER "test-cv")
target_link_libraries(
    testShikuang
    lifurencv
)
