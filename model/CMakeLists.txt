project(lifurenmodel VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/source/Model.cpp
    src/source/Layers.cpp
    src/source/Tensors.cpp
    src/source/Datasets.cpp
    src/source/model/Poetry.cpp
    src/source/layers/Conv2d.cpp
    src/source/layers/Linear.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "module")

target_link_libraries(
    ${PROJECT_NAME}
    lifurencore
    ggml
    ${OpenCV_LIBS}
)

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
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
