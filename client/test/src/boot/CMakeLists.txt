add_executable(testBoot TestBoot.cpp)
add_test(NAME testBoot COMMAND testBoot)
set_target_properties(testBoot PROPERTIES FOLDER "test/boot")
target_link_libraries(
    testBoot
    lifurenboot
)

add_executable(testDate TestDate.cpp)
add_test(NAME testDate COMMAND testDate)
set_target_properties(testDate PROPERTIES FOLDER "test/boot")
target_link_libraries(
    testDate
    lifurenboot
)

add_executable(testFile TestFile.cpp)
add_test(NAME testFile COMMAND testFile)
set_target_properties(testFile PROPERTIES FOLDER "test/boot")
target_link_libraries(
    testFile
    lifurenboot
)

add_executable(testString TestString.cpp)
add_test(NAME testString COMMAND testString)
set_target_properties(testString PROPERTIES FOLDER "test/boot")
target_link_libraries(
    testString
    lifurenboot
)
