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
