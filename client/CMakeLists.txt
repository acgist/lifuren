project(lifurenclient VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/RestClient.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "module")

if(CMAKE_HOST_WIN32)
    target_link_libraries(
        ${PROJECT_NAME}
        lifurencore
        httplib::httplib
    )
else()
    target_link_libraries(
        ${PROJECT_NAME}
        lifurencore
        $<IF:$<TARGET_EXISTS:cpp-httplib>,cpp-httplib,httplib::httplib>
    )
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${HEADER_DIR})

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(DIRECTORY ${HEADER_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
