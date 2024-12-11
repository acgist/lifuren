project(lifurenmodel VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/Layer.cpp
    src/Torch.cpp
    src/Model.cpp
    src/Tensor.cpp
    src/Dataset.cpp
    src/OnnxRuntime.cpp
    src/dataset/RawDataset.cpp
    src/dataset/FileDataset.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "module")

if(CMAKE_HOST_WIN32)
    target_link_libraries(
        ${PROJECT_NAME}
        lifurencore
        ${TORCH_LIBRARIES}
        ${ONNXRUNTIME_LIBS}
    )
else()
    target_link_libraries(
        ${PROJECT_NAME}
        lifurencore
        ${TORCH_LIBRARIES}
        onnxruntime::onnxruntime
    )
endif()

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC ${HEADER_DIR}
)

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
