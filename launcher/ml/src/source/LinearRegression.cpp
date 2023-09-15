#include "../header/MLPack.hpp"

void lifuren::linearRegression() {
    try {
        arma::mat    xTrain = arma::randu<arma::mat>(100, 5);
        arma::rowvec yTrain = arma::randu<arma::rowvec>(5);
        mlpack::regression::LinearRegression linear(xTrain, yTrain);
        arma::mat    xPred = arma::randu<arma::mat>(100, 5);
        arma::rowvec yPred;
        linear.Predict(xPred, yPred);
    } catch(const std::exception& e) {
        LOG(ERROR) << e.what();
    }
    
}