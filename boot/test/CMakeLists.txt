project(lifurentest VERSION 1.0.0 LANGUAGES C CXX)

aux_source_directory(${SOURCE_DIR} SOURCE_FILES)

file(GLOB SUB_SOURCE_FILES "src/source/*/*.cpp")

add_executable(${PROJECT_NAME} src/Main.cpp ${SOURCE_FILES} ${SUB_SOURCE_FILES})

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    lifurencore
    ${OpenCV_LIBS}
    ${FLTK_LIBRARIES}
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HEADER_DIR}
)

set(
    HEADER_FILES
    ${HEADER_DIR}/Core.hpp
    ${HEADER_DIR}/Fltk.hpp
    ${HEADER_DIR}/OpenCV.hpp
    ${HEADER_DIR}/LibTorch.hpp
)

install(FILES ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME})

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)

if(CMAKE_HOST_WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRT")

    install(FILES ${FLTK_DLLS}   DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${TORCH_DLLS}  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${OPENCV_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FLTK_DLLS}   $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}  $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()

# 测试

add_executable(testCore src/TestCore.cpp src/source/Core.cpp)
add_test(NAME testCore COMMAND testCore)
target_link_libraries(
    testCore PUBLIC
    lifurencore
)

add_executable(testCoreDates src/TestCoreDates.cpp)
add_test(NAME testCoreDates COMMAND testCoreDates)
target_link_libraries(
    testCoreDates PUBLIC
    lifurencore
)

add_executable(testCoreFiles src/TestCoreFiles.cpp)
add_test(NAME testCoreFiles COMMAND testCoreFiles)
target_link_libraries(
    testCoreFiles PUBLIC
    lifurencore
)

add_executable(testCorePoetry src/TestCorePoetry.cpp)
add_test(NAME testCorePoetry COMMAND testCorePoetry)
target_link_libraries(
    testCorePoetry PUBLIC
    lifurencore
)

add_executable(testCoreCollections src/TestCoreCollections.cpp)
add_test(NAME testCoreCollections COMMAND testCoreCollections)
target_link_libraries(
    testCoreCollections PUBLIC
    lifurencore
)

add_executable(testFltk src/TestFltk.cpp src/source/fltk/Fltk.cpp)
add_test(NAME testFltk COMMAND testFltk)
target_link_libraries(
    testFltk PUBLIC
    lifurencore
    ${FLTK_LIBRARIES}
)

add_executable(testLibTorchCUDA src/TestLibTorchCUDA.cpp src/source/libtorch/CUDA.cpp)
add_test(NAME testLibTorchCUDA COMMAND testLibTorchCUDA)
target_link_libraries(
    testLibTorchCUDA PUBLIC
    lifurencore
)

add_executable(testLibTorchTensor src/TestLibTorchTensor.cpp src/source/libtorch/Tensor.cpp)
add_test(NAME testLibTorchTensor COMMAND testLibTorchTensor)
target_link_libraries(
    testLibTorchTensor PUBLIC
    lifurencore
)

add_executable(testOpenCV src/TestOpenCV.cpp src/source/opencv/OpenCV.cpp)
add_test(NAME testOpenCV COMMAND testOpenCV)
target_link_libraries(
    testOpenCV PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)

add_executable(testOpenCVFace src/TestOpenCVFace.cpp src/source/opencv/OpenCV.cpp src/source/opencv/Face.cpp)
add_test(NAME testOpenCVFace COMMAND testOpenCVFace)
target_link_libraries(
    testOpenCVFace PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)

add_executable(testOpenCVColor src/TestOpenCVColor.cpp src/source/opencv/OpenCV.cpp src/source/opencv/Color.cpp)
add_test(NAME testOpenCVColor COMMAND testOpenCVColor)
target_link_libraries(
    testOpenCVColor PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)

add_executable(testOpenCVResize src/TestOpenCVResize.cpp src/source/opencv/OpenCV.cpp src/source/opencv/Resize.cpp)
add_test(NAME testOpenCVResize COMMAND testOpenCVResize)
target_link_libraries(
    testOpenCVResize PUBLIC
    lifurencore
    ${OpenCV_LIBS}
)
