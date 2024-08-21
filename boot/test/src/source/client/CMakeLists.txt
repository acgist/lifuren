add_executable(testRestClient TestRestClient.cpp)
add_test(NAME testRestClient COMMAND testRestClient)
target_link_libraries(
    testRestClient
    lifurenclient
)

add_executable(testCommandClient TestCommandClient.cpp)
add_test(NAME testCommandClient COMMAND testCommandClient)
target_link_libraries(
    testCommandClient
    lifurenclient
)
