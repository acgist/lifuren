project(lifurentest VERSION 1.0.0 LANGUAGES C CXX)

aux_source_directory(${SOURCE_DIR} SOURCE_FILES)

file(GLOB SUB_SOURCE_FILES "src/source/*/*.cpp")

add_executable(${PROJECT_NAME} src/Main.cpp ${SOURCE_FILES} ${SUB_SOURCE_FILES})

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    lifurencore
    LAPACK::LAPACK
    ${OpenCV_LIBS}
    ${FLTK_LIBRARIES}
    ${ARMADILLO_LIBRARY}
    # OpenBLAS::OpenBLAS
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HEADER_DIR}
)

set(
    HEADER_FILES
    ${HEADER_DIR}/CNN.hpp
    ${HEADER_DIR}/RNN.hpp
    ${HEADER_DIR}/GAN.hpp
    ${HEADER_DIR}/Boot.hpp
    ${HEADER_DIR}/Fltk.hpp
    ${HEADER_DIR}/Mnist.hpp
    ${HEADER_DIR}/OpenCV.hpp
    ${HEADER_DIR}/MLPack.hpp
    ${HEADER_DIR}/LibTorch.hpp
    ${HEADER_DIR}/Transformer.hpp
)

install(FILES ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)

add_definitions(-DC10_USE_GLOG)

if(CMAKE_HOST_WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${FLTK_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${MLPACK_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENCV_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()

# 测试

add_executable(testBoot src/TestBoot.cpp src/source/Boot.cpp)
add_test(NAME testBoot COMMAND testBoot)
target_link_libraries(
    testBoot PUBLIC
    lifurencore
)

add_executable(testFltk src/TestFltk.cpp src/source/fltk/Fltk.cpp)
add_test(NAME testFltk COMMAND testFltk)
target_link_libraries(
    testFltk PUBLIC
    lifurencore
    ${FLTK_LIBRARIES}
)

add_executable(testOpenCV src/TestOpenCV.cpp src/source/opencv/OpenCV.cpp)
add_test(NAME testOpenCV COMMAND testOpenCV)
target_link_libraries(
    testOpenCV PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)

add_executable(testOpenCVColor src/TestOpenCVColor.cpp src/source/opencv/OpenCV.cpp src/source/opencv/OpenCVColor.cpp)
add_test(NAME testOpenCVColor COMMAND testOpenCVColor)
target_link_libraries(
    testOpenCVColor PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)

add_executable(testOpenCVFace src/TestOpenCVFace.cpp src/source/opencv/OpenCV.cpp src/source/opencv/OpenCVFace.cpp)
add_test(NAME testOpenCVFace COMMAND testOpenCVFace)
target_link_libraries(
    testOpenCVFace PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)

add_executable(testOpenCVResize src/TestOpenCVResize.cpp src/source/opencv/OpenCV.cpp src/source/opencv/OpenCVResize.cpp)
add_test(NAME testOpenCVResize COMMAND testOpenCVResize)
target_link_libraries(
    testOpenCVResize PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)

add_executable(testMLPackMatrix src/TestMLPackMatrix.cpp src/source/mlpack/Matrix.cpp)
add_test(NAME testMLPackMatrix COMMAND testMLPackMatrix)
target_link_libraries(
    testMLPackMatrix PUBLIC
    lifurencore
    LAPACK::LAPACK
    ${ARMADILLO_LIBRARY}
    # OpenBLAS::OpenBLAS
)

add_executable(testMLPackLoadFile src/TestMLPackLoadFile.cpp src/source/mlpack/LoadFile.cpp)
add_test(NAME testMLPackLoadFile COMMAND testMLPackLoadFile)
target_link_libraries(
    testMLPackLoadFile PUBLIC
    lifurencore
    LAPACK::LAPACK
    ${ARMADILLO_LIBRARY}
    # OpenBLAS::OpenBLAS
)

add_executable(testMLPackLinearRegression src/TestMLPackLinearRegression.cpp src/source/mlpack/LinearRegression.cpp)
add_test(NAME testMLPackLinearRegression COMMAND testMLPackLinearRegression)
target_link_libraries(
    testMLPackLinearRegression PUBLIC
    lifurencore
    LAPACK::LAPACK
    ${ARMADILLO_LIBRARY}
    # OpenBLAS::OpenBLAS
)

add_executable(testMLPackSoftmaxRegression src/TestMLPackSoftmaxRegression.cpp src/source/mlpack/SoftmaxRegression.cpp)
add_test(NAME testMLPackSoftmaxRegression COMMAND testMLPackSoftmaxRegression)
target_link_libraries(
    testMLPackSoftmaxRegression PUBLIC
    lifurencore
    LAPACK::LAPACK
    ${ARMADILLO_LIBRARY}
    # OpenBLAS::OpenBLAS
)

add_executable(testMLPackLogisticRegression src/TestMLPackLogisticRegression.cpp src/source/mlpack/LogisticRegression.cpp)
add_test(NAME testMLPackLogisticRegression COMMAND testMLPackLogisticRegression)
target_link_libraries(
    testMLPackLogisticRegression PUBLIC
    lifurencore
    LAPACK::LAPACK
    ${ARMADILLO_LIBRARY}
    # OpenBLAS::OpenBLAS
)

add_executable(testLibTorchMnist src/TestLibTorchMnist.cpp src/source/libtorch/Mnist.cpp)
add_test(NAME testLibTorchMnist COMMAND testLibTorchMnist)
target_link_libraries(
    testLibTorchMnist PUBLIC
    lifurencore
)

add_executable(testLibTorchLayer src/TestLibTorchLayer.cpp)
add_test(NAME testLibTorchLayer COMMAND testLibTorchLayer)
target_link_libraries(
    testLibTorchLayer PUBLIC
    lifurencore
)

add_executable(testLibTorchTensor src/TestLibTorchTensor.cpp src/source/libtorch/Tensor.cpp)
add_test(NAME testLibTorchTensor COMMAND testLibTorchTensor)
target_link_libraries(
    testLibTorchTensor PUBLIC
    lifurencore
)

add_executable(testLibTorchLinearRegression src/TestLibTorchLinearRegression.cpp src/source/libtorch/LinearRegression.cpp)
add_test(NAME testLibTorchLinearRegression COMMAND testLibTorchLinearRegression)
target_link_libraries(
    testLibTorchLinearRegression PUBLIC
    lifurencore
)

add_executable(testLibTorchActivationFunction src/TestLibTorchActivationFunction.cpp src/source/libtorch/ActivationFunction.cpp)
add_test(NAME testLibTorchActivationFunction COMMAND testLibTorchActivationFunction)
target_link_libraries(
    testLibTorchActivationFunction PUBLIC
    lifurencore
)
