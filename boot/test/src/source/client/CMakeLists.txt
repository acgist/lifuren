add_executable(testRestClient TestRestClient.cpp)
add_test(NAME testRestClient COMMAND testRestClient)
set_target_properties(testRestClient PROPERTIES FOLDER "test-client")
target_link_libraries(
    testRestClient
    lifurenclient
)

add_executable(testCommandClient TestCommandClient.cpp)
add_test(NAME testCommandClient COMMAND testCommandClient)
set_target_properties(testCommandClient PROPERTIES FOLDER "test-client")
target_link_libraries(
    testCommandClient
    lifurenclient
)
