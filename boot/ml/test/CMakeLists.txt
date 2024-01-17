add_executable(testBoot ./TestBoot.cpp ../src/source/Boot.cpp)
add_test(NAME testBoot COMMAND testBoot)
target_link_libraries(
    testBoot PUBLIC
    lifurencore
)

add_executable(testFltk ./TestFltk.cpp ../src/source/fltk/Fltk.cpp)
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

add_executable(testOpenCVColor ./TestOpenCVColor.cpp ../src/source/opencv/OpenCV.cpp ../src/source/opencv/OpenCVColor.cpp)
add_test(NAME testOpenCVColor COMMAND testOpenCVColor)
target_link_libraries(
    testOpenCVColor PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)

add_executable(testOpenCVFace ./TestOpenCVFace.cpp ../src/source/opencv/OpenCV.cpp ../src/source/opencv/OpenCVFace.cpp)
add_test(NAME testOpenCVFace COMMAND testOpenCVFace)
target_link_libraries(
    testOpenCVFace PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)

add_executable(testOpenCVResize ./TestOpenCVResize.cpp ../src/source/opencv/OpenCV.cpp ../src/source/opencv/OpenCVResize.cpp)
add_test(NAME testOpenCVResize COMMAND testOpenCVResize)
target_link_libraries(
    testOpenCVResize PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)

add_executable(testMLPackMatrix ./TestMLPackMatrix.cpp ../src/source/mlpack/Matrix.cpp)
add_test(NAME testMLPackMatrix COMMAND testMLPackMatrix)
target_link_libraries(
    testMLPackMatrix PUBLIC
    lifurencore
    LAPACK::LAPACK
    ${ARMADILLO_LIBRARY}
    # OpenBLAS::OpenBLAS
)

add_executable(testMLPackLoadFile ./TestMLPackLoadFile.cpp ../src/source/mlpack/LoadFile.cpp)
add_test(NAME testMLPackLoadFile COMMAND testMLPackLoadFile)
target_link_libraries(
    testMLPackLoadFile PUBLIC
    lifurencore
    LAPACK::LAPACK
    ${ARMADILLO_LIBRARY}
    # OpenBLAS::OpenBLAS
)

add_executable(testMLPackLinearRegression ./TestMLPackLinearRegression.cpp ../src/source/mlpack/LinearRegression.cpp)
add_test(NAME testMLPackLinearRegression COMMAND testMLPackLinearRegression)
target_link_libraries(
    testMLPackLinearRegression PUBLIC
    lifurencore
    LAPACK::LAPACK
    ${ARMADILLO_LIBRARY}
    # OpenBLAS::OpenBLAS
)

add_executable(testMLPackSoftmaxRegression ./TestMLPackSoftmaxRegression.cpp ../src/source/mlpack/SoftmaxRegression.cpp)
add_test(NAME testMLPackSoftmaxRegression COMMAND testMLPackSoftmaxRegression)
target_link_libraries(
    testMLPackSoftmaxRegression PUBLIC
    lifurencore
    LAPACK::LAPACK
    ${ARMADILLO_LIBRARY}
    # OpenBLAS::OpenBLAS
)

add_executable(testMLPackLogisticRegression ./TestMLPackLogisticRegression.cpp ../src/source/mlpack/LogisticRegression.cpp)
add_test(NAME testMLPackLogisticRegression COMMAND testMLPackLogisticRegression)
target_link_libraries(
    testMLPackLogisticRegression PUBLIC
    lifurencore
    LAPACK::LAPACK
    ${ARMADILLO_LIBRARY}
    # OpenBLAS::OpenBLAS
)

add_executable(testLibTorchMnist ./TestLibTorchMnist.cpp ../src/source/libtorch/Mnist.cpp)
add_test(NAME testLibTorchMnist COMMAND testLibTorchMnist)
target_link_libraries(
    testLibTorchMnist PUBLIC
    lifurencore
)

add_executable(testLibTorchLayer ./TestLibTorchLayer.cpp)
add_test(NAME testLibTorchLayer COMMAND testLibTorchLayer)
target_link_libraries(
    testLibTorchLayer PUBLIC
    lifurencore
)

add_executable(testLibTorchTensor ./TestLibTorchTensor.cpp ../src/source/libtorch/Tensor.cpp)
add_test(NAME testLibTorchTensor COMMAND testLibTorchTensor)
target_link_libraries(
    testLibTorchTensor PUBLIC
    lifurencore
)

add_executable(testLibTorchLinearRegression ./TestLibTorchLinearRegression.cpp ../src/source/libtorch/LinearRegression.cpp)
add_test(NAME testLibTorchLinearRegression COMMAND testLibTorchLinearRegression)
target_link_libraries(
    testLibTorchLinearRegression PUBLIC
    lifurencore
)

add_executable(testLibTorchActivationFunction ./TestLibTorchActivationFunction.cpp ../src/source/libtorch/ActivationFunction.cpp)
add_test(NAME testLibTorchActivationFunction COMMAND testLibTorchActivationFunction)
target_link_libraries(
    testLibTorchActivationFunction PUBLIC
    lifurencore
)
