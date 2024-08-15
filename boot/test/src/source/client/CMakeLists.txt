add_executable(testRestClient TestRestClient.cpp)
add_test(NAME testRestClient COMMAND testRestClient)
target_link_libraries(
    testRestClient
    lifurenclient
)
