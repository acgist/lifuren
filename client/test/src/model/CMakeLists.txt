
add_executable(testWudaozi TestWudaozi.cpp)
add_test(NAME testWudaozi COMMAND testWudaozi)
set_target_properties(testWudaozi PROPERTIES FOLDER "test/model")
target_link_libraries(
    testWudaozi
    lifurencore
)

add_executable(testClassify TestClassify.cpp)
add_test(NAME testClassify COMMAND testClassify)
set_target_properties(testClassify PROPERTIES FOLDER "test/model")
target_link_libraries(
    testClassify
    lifurencore
)