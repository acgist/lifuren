project(lifurenml VERSION 1.0.0 LANGUAGES C CXX)

add_executable(
    ${PROJECT_NAME}
    src/ML.cpp
    src/source/LifurenGG.cpp
    src/source/3DPointCloud.cpp
)

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    lifurengg
)
