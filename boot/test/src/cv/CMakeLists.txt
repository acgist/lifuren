add_executable(testImage TestImage.cpp)
add_test(NAME testImage COMMAND testImage)
set_target_properties(testImage PROPERTIES FOLDER "test-cv")
target_link_libraries(
    testImage
    lifurencv
)

add_executable(testCVModel TestCVModel.cpp)
add_test(NAME testCVModel COMMAND testCVModel)
set_target_properties(testCVModel PROPERTIES FOLDER "test-cv")
target_link_libraries(
    testCVModel
    lifurencv
)

