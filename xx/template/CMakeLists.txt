cmake_minimum_required(VERSION 3.0.0)

project(lifurentemplate VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -std=c17 -O0 -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -std=c17 -O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++17 -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17 -O3")

if(CMAKE_HOST_UNIX)
    # include_directories()
elseif(CMAKE_HOST_WIN32)
    # include_directories(D:/develop/libtorch/include)
    # include_directories(D:/develop/libtorch/include/torch/csrc/api/include/torch)
else()
    # message("不支持的系统")
endif()

set(WITH_GTEST  OFF)
set(WITH_GFLAGS OFF)

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

add_subdirectory("../../deps/glog" glog)

enable_testing()
add_subdirectory(test)

set(HEADER_DIR src/header/)
set(SOURCE_DIR src/source/)

include_directories(${HEADER_DIR})

aux_source_directory(${SOURCE_DIR} SOURCE_FILES)

add_executable(${PROJECT_NAME} "src/Main.cpp" ${SOURCE_FILES})

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

set(
    HEADER_FILES
    "${HEADER_DIR}/Template.hpp"
)

install(FILES ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    glog::glog
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HEADER_DIR}
)
