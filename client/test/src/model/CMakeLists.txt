add_executable(testChopin TestChopin.cpp)
add_test(NAME testChopin COMMAND testChopin)
set_target_properties(testChopin PROPERTIES FOLDER "test-model")
target_link_libraries(
    testChopin
    lifurencore
)

add_executable(testMozart TestMozart.cpp)
add_test(NAME testMozart COMMAND testMozart)
set_target_properties(testMozart PROPERTIES FOLDER "test-model")
target_link_libraries(
    testMozart
    lifurencore
)

add_executable(testShikuang TestShikuang.cpp)
add_test(NAME testShikuang COMMAND testShikuang)
set_target_properties(testShikuang PROPERTIES FOLDER "test-model")
target_link_libraries(
    testShikuang
    lifurencore
)
