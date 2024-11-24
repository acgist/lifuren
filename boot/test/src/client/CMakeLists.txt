add_executable(testRestClient TestRestClient.cpp)
add_test(NAME testRestClient COMMAND testRestClient)
set_target_properties(testRestClient PROPERTIES FOLDER "test-client")
target_link_libraries(
    testRestClient
    lifurenclient
)
