project(lifurentest VERSION 1.0.0 LANGUAGES C CXX)

add_executable(
    ${PROJECT_NAME}
    src/Test.cpp
)

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    lifurencc
)
