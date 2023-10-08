#include "../header/MLPack.hpp"

void lifuren::testMLPackLinearRegression() {
    try {
        arma::mat    xTrain = arma::randu<arma::mat>(100, 5);
        arma::rowvec yTrain = arma::randu<arma::rowvec>(5);
        LOG(INFO) << 1;
        mlpack::regression::LinearRegression linear(xTrain, yTrain);
        arma::mat    xPred = arma::randu<arma::mat>(100, 5);
        arma::rowvec yPred;
        LOG(INFO) << 2;
        linear.Predict(xPred, yPred);
        LOG(INFO) << 3;
        LOG(INFO) << yPred;
    } catch(const std::exception& e) {
        LOG(ERROR) << e.what();
    }   
}