#include "../header/MLPack.hpp"
#include <iostream>

void lifuren::linearRegression() {
    std::cout << "123" << std::endl;
    arma::mat x(50, 1, arma::fill::randu);
    std::cout << "123" << std::endl;
    arma::rowvec y = x + arma::randu(50) * 0.1;
    std::cout << "123" << std::endl;
    mlpack::LinearRegression linear(x, y);
    std::cout << "123" << std::endl;
    arma::mat testX(10, 2, arma::fill::randu);
    std::cout << "123" << std::endl;
    arma::rowvec predY;
    std::cout << testX << std::endl;
    linear.Predict(testX, predY);
    std::cout << predY << std::endl;
}