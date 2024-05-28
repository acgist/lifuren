add_executable(testFace TestFace.cpp)
add_test(NAME testFace COMMAND testFace)
target_link_libraries(
    testFace PUBLIC
    lifurencore
)

add_executable(testColor TestColor.cpp)
add_test(NAME testColor COMMAND testColor)
target_link_libraries(
    testColor PUBLIC
    lifurencore
)

add_executable(testResize TestResize.cpp)
add_test(NAME testResize COMMAND testResize)
target_link_libraries(
    testResize PUBLIC
    lifurencore
)
