project(lifurenml VERSION 1.0.0 LANGUAGES C CXX)

add_executable(
    ${PROJECT_NAME}
    src/Main.cpp
    src/source/3DPointCloud.cpp
    src/source/AssignmentLearning.cpp
    src/source/Base64Encoder.cpp
    src/source/LifurenGG.cpp
    src/source/LinearRegression.cpp
    src/source/MathGL.cpp
)

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    lifurengg
)
