add_subdirectory(main)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  message(STATUS "Debug with test")
  add_subdirectory(test)
else()
  message(STATUS "Release without test")
endif()
