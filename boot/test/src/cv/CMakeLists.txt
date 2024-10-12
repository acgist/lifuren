add_executable(testImage TestImage.cpp)
add_test(NAME testImage COMMAND testImage)
set_target_properties(testImage PROPERTIES FOLDER "test-cv")
target_link_libraries(
    testImage
    lifurencv
)

add_executable(testStableDiffusionCPPPaintClient TestStableDiffusionCPPPaintClient.cpp)
add_test(NAME testStableDiffusionCPPPaintClient COMMAND testStableDiffusionCPPPaintClient)
set_target_properties(testStableDiffusionCPPPaintClient PROPERTIES FOLDER "test-cv")
target_link_libraries(
    testStableDiffusionCPPPaintClient
    lifurencv
)
