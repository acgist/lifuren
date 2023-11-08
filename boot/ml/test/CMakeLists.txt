add_executable(testBoot ./TestBoot.cpp ../src/source/Boot.cpp)
add_test(NAME testBoot COMMAND testBoot)
target_link_libraries(
    testBoot PUBLIC
    lifurencore
)

add_executable(testFltk ./testFltk.cpp ../src/source/fltk/Fltk.cpp)
add_test(NAME testFltk COMMAND testFltk)
target_link_libraries(
    testFltk PUBLIC
    lifurencore
    ${FLTK_LIBRARIES}
)

add_executable(testOpenCV ./TestOpenCV.cpp ../src/source/opencv/OpenCV.cpp)
add_test(NAME testOpenCV COMMAND testOpenCV)
target_link_libraries(
    testOpenCV PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)

add_executable(testLoadFile ./TestLoadFile.cpp ../src/source/mlpack/LoadFile.cpp)
add_test(NAME testLoadFile COMMAND testLoadFile)
target_link_libraries(
    testLoadFile PUBLIC
    lifurencore
)

add_executable(testMLPackMatrix ./TestMLPackMatrix.cpp ../src/source/mlpack/Matrix.cpp)
add_test(NAME testMLPackMatrix COMMAND testMLPackMatrix)
target_link_libraries(
    testMLPackMatrix PUBLIC
    lifurencore
)

add_executable(testMLPackLinearRegression ./TestMLPackLinearRegression.cpp ../src/source/mlpack/LinearRegression.cpp)
add_test(NAME testMLPackLinearRegression COMMAND testMLPackLinearRegression)
target_link_libraries(
    testMLPackLinearRegression PUBLIC
    lifurencore
)

add_executable(testMLPackLogisticRegression ./TestMLPackLogisticRegression.cpp ../src/source/mlpack/LogisticRegression.cpp)
add_test(NAME testMLPackLogisticRegression COMMAND testMLPackLogisticRegression)
target_link_libraries(
    testMLPackLogisticRegression PUBLIC
    lifurencore
)

add_executable(testMLPackSoftmaxRegression ./TestMLPackSoftmaxRegression.cpp ../src/source/mlpack/SoftmaxRegression.cpp)
add_test(NAME testMLPackSoftmaxRegression COMMAND testMLPackSoftmaxRegression)
target_link_libraries(
    testMLPackSoftmaxRegression PUBLIC
    lifurencore
)

add_executable(testMatrix ./TestMatrix.cpp ../src/source/libtorch/Matrix.cpp)
add_test(NAME testMatrix COMMAND testMatrix)
target_link_libraries(
    testMatrix PUBLIC
    lifurencore
)

add_executable(testActivationFunction ./TestActivationFunction.cpp ../src/source/libtorch/ActivationFunction.cpp)
add_test(NAME testActivationFunction COMMAND testActivationFunction)
target_link_libraries(
    testActivationFunction PUBLIC
    lifurencore
)

add_executable(testLinearRegression ./TestLinearRegression.cpp ../src/source/libtorch/LinearRegression.cpp)
add_test(NAME testLinearRegression COMMAND testLinearRegression)
target_link_libraries(
    testLinearRegression PUBLIC
    lifurencore
)
