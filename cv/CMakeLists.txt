project(lifurencv VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/Audio.cpp
    src/Image.cpp
    src/Video.cpp
    src/OpenCV.cpp
    src/ActClient.cpp
    src/PaintClient.cpp
    src/ComposeClient.cpp
    src/act/GuanhanqinActClient.cpp
    src/act/TangxianzuActClient.cpp
    src/paint/CycleGANPaintClient.cpp
    src/paint/StyleGANPaintClient.cpp
    src/compose/ShikuangComposeClient.cpp
    src/compose/LiguinianComposeClient.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "module")

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    lifurenmodel
    lifurenclient
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
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
