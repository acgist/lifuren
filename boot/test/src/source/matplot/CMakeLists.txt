add_executable(testMatplot TestMatplot.cpp)
add_test(NAME testMatplot COMMAND testMatplot)
target_link_libraries(
    testMatplot PUBLIC
    Matplot++::cimg Matplot++::matplot
)
