project(lifurenlibtorch VERSION 1.0.0 LANGUAGES C CXX)

set(HEADER_DIR src/header/)
set(SOURCE_DIR src/source/)

find_package(Torch REQUIRED)

aux_source_directory(${SOURCE_DIR} SOURCE_FILES)

add_executable(${PROJECT_NAME} src/Main.cpp ${SOURCE_FILES})

add_definitions(-DC10_USE_GLOG)

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    lifurengg
    ${TORCH_LIBRARIES}
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HEADER_DIR}
)

set(
    HEADER_FILES
    ${HEADER_DIR}/LibTorch.hpp
)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)

install(
    FILES
    ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)