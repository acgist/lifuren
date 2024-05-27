add_executable(testOpenCVFace TestOpenCVFace.cpp Face.cpp)
add_test(NAME testOpenCVFace COMMAND testOpenCVFace)
target_link_libraries(
    testOpenCVFace PUBLIC
    lifurencore
)

add_executable(testOpenCVColor TestOpenCVColor.cpp Color.cpp)
add_test(NAME testOpenCVColor COMMAND testOpenCVColor)
target_link_libraries(
    testOpenCVColor PUBLIC
    lifurencore
)

add_executable(testOpenCVResize TestOpenCVResize.cpp Resize.cpp)
add_test(NAME testOpenCVResize COMMAND testOpenCVResize)
target_link_libraries(
    testOpenCVResize PUBLIC
    lifurencore
)
