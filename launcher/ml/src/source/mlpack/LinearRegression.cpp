#include "../header/MLPack.hpp"

void lifuren::testMLPackLinearRegression() {
    try {
        arma::mat xTrain = arma::randu<arma::mat>(2, 5);
        // arma::mat xTrain = arma::randu<arma::mat>(10, 5);
        // arma::rowvec yTrain = arma::randu<arma::rowvec>(5);
        arma::rowvec yTrain = arma::sum(xTrain) + 1;
        LOG(INFO) << std::endl << xTrain;
        LOG(INFO) << std::endl << yTrain;
        mlpack::regression::LinearRegression linear(xTrain, yTrain);
        arma::mat xPred = arma::randu<arma::mat>(2, 5);
        // arma::mat xPred = arma::randu<arma::mat>(10, 5);
        arma::rowvec yPred;
        linear.Predict(xPred, yPred);
        LOG(INFO) << std::endl << xPred;
        LOG(INFO) << std::endl << yPred;
    } catch(const std::exception& e) {
        LOG(ERROR) << e.what();
    }   
}