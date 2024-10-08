add_executable(testImages TestImages.cpp)
add_test(NAME testImages COMMAND testImages)
set_target_properties(testImages PROPERTIES FOLDER "test-cv")
target_link_libraries(
    testImages
    lifurencv
)

add_executable(testStableDiffusionCPPPaintClient TestStableDiffusionCPPPaintClient.cpp)
add_test(NAME testStableDiffusionCPPPaintClient COMMAND testStableDiffusionCPPPaintClient)
set_target_properties(testStableDiffusionCPPPaintClient PROPERTIES FOLDER "test-cv")
target_link_libraries(
    testStableDiffusionCPPPaintClient
    lifurencv
)
