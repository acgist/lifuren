add_executable(testDates TestDates.cpp)
add_test(NAME testDates COMMAND testDates)
set_target_properties(testDates PROPERTIES FOLDER "test")
target_link_libraries(
    testDates
    lifurencore
)

add_executable(testFiles TestFiles.cpp)
add_test(NAME testFiles COMMAND testFiles)
set_target_properties(testFiles PROPERTIES FOLDER "test")
target_link_libraries(
    testFiles
    lifurencore
)

add_executable(testJsons TestJsons.cpp)
add_test(NAME testJsons COMMAND testJsons)
set_target_properties(testJsons PROPERTIES FOLDER "test")
target_link_libraries(
    testJsons
    lifurencore
)

add_executable(testYamls TestYamls.cpp)
add_test(NAME testYamls COMMAND testYamls)
set_target_properties(testYamls PROPERTIES FOLDER "test")
target_link_libraries(
    testYamls
    lifurencore
)

add_executable(testConfig TestConfig.cpp)
add_test(NAME testConfig COMMAND testConfig)
set_target_properties(testConfig PROPERTIES FOLDER "test")
target_link_libraries(
    testConfig
    lifurencore
)

add_executable(testPoetry TestPoetry.cpp)
add_test(NAME testPoetry COMMAND testPoetry)
set_target_properties(testPoetry PROPERTIES FOLDER "test")
target_link_libraries(
    testPoetry
    lifurencore
)

add_executable(testStrings TestStrings.cpp)
add_test(NAME testStrings COMMAND testStrings)
set_target_properties(testStrings PROPERTIES FOLDER "test")
target_link_libraries(
    testStrings
    lifurencore
)

add_executable(testLifuren TestLifuren.cpp)
add_test(NAME testLifuren COMMAND testLifuren)
set_target_properties(testLifuren PROPERTIES FOLDER "test")
target_link_libraries(
    testLifuren
    lifurencore
)

add_executable(testException TestException.cpp)
add_test(NAME testException COMMAND testException)
set_target_properties(testException PROPERTIES FOLDER "test")
target_link_libraries(
    testException
    lifurencore
)

add_executable(testCollections TestCollections.cpp)
add_test(NAME testCollections COMMAND testCollections)
set_target_properties(testCollections PROPERTIES FOLDER "test")
target_link_libraries(
    testCollections
    lifurencore
)
