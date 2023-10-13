#include "../../header/MLPack.hpp"
// #include <mlpack/core.hpp>
// #include <mlpack/methods/linear_regression.hpp>

void lifuren::testMLPackLinearRegression() {
    try {
        arma::mat xTrain = arma::randu<arma::mat>(2, 5);
        // arma::mat xTrain = arma::randu<arma::mat>(10, 5);
        // arma::rowvec yTrain = arma::randu<arma::rowvec>(5);
        arma::rowvec rand   = arma::randu<arma::rowvec>(5);
        // arma::rowvec yTrain = arma::sum(xTrain) + 1;
        arma::rowvec yTrain = arma::sum(xTrain) + rand;
        LOG(INFO) << std::endl << xTrain;
        LOG(INFO) << std::endl << yTrain;
        mlpack::regression::LinearRegression linear(xTrain, yTrain);
        // mlpack::regression::LinearRegression linear(xTrain, yTrain, 0.0001);
        arma::mat xPred = arma::randu<arma::mat>(2, 5);
        // arma::mat xPred = arma::randu<arma::mat>(10, 5);
        arma::rowvec yPred;
        linear.Predict(xPred, yPred);
        LOG(INFO) << std::endl << xPred;
        LOG(INFO) << std::endl << yPred;
        // 差值
        LOG(INFO) << std::endl << rand;
        LOG(INFO) << std::endl << yPred - arma::sum(xPred);
    } catch(const std::exception& e) {
        LOG(ERROR) << e.what();
    }
}