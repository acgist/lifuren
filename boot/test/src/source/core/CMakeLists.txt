add_executable(testDates TestDates.cpp)
add_test(NAME testDates COMMAND testDates)
target_link_libraries(
    testDates PUBLIC
    lifurencore
)

add_executable(testFiles TestFiles.cpp)
add_test(NAME testFiles COMMAND testFiles)
target_link_libraries(
    testFiles PUBLIC
    lifurencore
)

add_executable(testJsons TestJsons.cpp)
add_test(NAME testJsons COMMAND testJsons)
target_link_libraries(
    testJsons PUBLIC
    lifurencore
)

add_executable(testYamls TestYamls.cpp)
add_test(NAME testYamls COMMAND testYamls)
target_link_libraries(
    testYamls PUBLIC
    lifurencore
)

add_executable(testPoetry TestPoetry.cpp)
add_test(NAME testPoetry COMMAND testPoetry)
target_link_libraries(
    testPoetry PUBLIC
    lifurencore
)

add_executable(testStrings TestStrings.cpp)
add_test(NAME testStrings COMMAND testStrings)
target_link_libraries(
    testStrings PUBLIC
    lifurencore
)

add_executable(testDatasets TestDatasets.cpp)
add_test(NAME testDatasets COMMAND testDatasets)
target_link_libraries(
    testDatasets PUBLIC
    lifurencore
)

add_executable(testCollections TestCollections.cpp)
add_test(NAME testCollections COMMAND testCollections)
target_link_libraries(
    testCollections PUBLIC
    lifurencore
)