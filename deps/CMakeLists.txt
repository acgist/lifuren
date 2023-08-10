set(WITH_GTEST  OFF)
set(WITH_GFLAGS OFF)

add_subdirectory(dlib)
add_subdirectory(glog)
add_subdirectory(json)

if(CMAKE_HOST_UNIX)
  add_subdirectory(opencv)
endif()
