add_executable(testDate TestDate.cpp)
add_test(NAME testDate COMMAND testDate)
set_target_properties(testDate PROPERTIES FOLDER "test-boot")
target_link_libraries(
    testDate
    lifurenboot
)

add_executable(testFile TestFile.cpp)
add_test(NAME testFile COMMAND testFile)
set_target_properties(testFile PROPERTIES FOLDER "test-boot")
target_link_libraries(
    testFile
    lifurenboot
)

add_executable(testConfig TestConfig.cpp)
add_test(NAME testConfig COMMAND testConfig)
set_target_properties(testConfig PROPERTIES FOLDER "test-boot")
target_link_libraries(
    testConfig
    lifurenboot
)

add_executable(testString TestString.cpp)
add_test(NAME testString COMMAND testString)
set_target_properties(testString PROPERTIES FOLDER "test-boot")
target_link_libraries(
    testString
    lifurenboot
)

add_executable(testThread TestThread.cpp)
add_test(NAME testThread COMMAND testThread)
set_target_properties(testThread PROPERTIES FOLDER "test-boot")
target_link_libraries(
    testThread
    lifurenboot
)
