add_executable(testBach TestBach.cpp)
add_test(NAME testBach COMMAND testBach)
set_target_properties(testBach PROPERTIES FOLDER "test-model")
target_link_libraries(
    testBach
    lifurencore
)

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

add_executable(testWudaozi TestWudaozi.cpp)
add_test(NAME testWudaozi COMMAND testWudaozi)
set_target_properties(testWudaozi PROPERTIES FOLDER "test-model")
target_link_libraries(
    testWudaozi
    lifurencore
)

add_executable(testShikuang TestShikuang.cpp)
add_test(NAME testShikuang COMMAND testShikuang)
set_target_properties(testShikuang PROPERTIES FOLDER "test-model")
target_link_libraries(
    testShikuang
    lifurencore
)

add_executable(testBeethoven TestBeethoven.cpp)
add_test(NAME testBeethoven COMMAND testBeethoven)
set_target_properties(testBeethoven PROPERTIES FOLDER "test-model")
target_link_libraries(
    testBeethoven
    lifurencore
)