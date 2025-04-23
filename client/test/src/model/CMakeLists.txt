add_executable(testWudaozi TestWudaozi.cpp)
add_test(NAME testWudaozi COMMAND testWudaozi)
set_target_properties(testWudaozi PROPERTIES FOLDER "test/model")
target_link_libraries(
    testWudaozi
    lifurencore
)

add_executable(testShikuang TestShikuang.cpp)
add_test(NAME testShikuang COMMAND testShikuang)
set_target_properties(testShikuang PROPERTIES FOLDER "test/model")
target_link_libraries(
    testShikuang
    lifurencore
)
