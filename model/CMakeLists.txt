project(lifurenmodel VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/Model.cpp
    src/Layers.cpp
    src/Dataset.cpp
    src/Tensors.cpp
    src/Datasets.cpp
    src/RawDataset.cpp
    src/FileDataset.cpp
    src/ShardingDataset.cpp
    src/layers/Conv2d.cpp
    src/layers/Linear.cpp
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
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lifuren/model)
