#include "../../header/MLPack.hpp"

void lifuren::testMLPackLinearRegression() {
    const arma::mat xTrain = arma::randu<arma::mat>(2, 5);
    const arma::rowvec rand   = arma::randu<arma::rowvec>(5);
    const arma::rowvec yTrain = arma::sum(xTrain) + rand;
    LOG(INFO) << "训练数据x" << std::endl << xTrain;
    LOG(INFO) << "训练数据y" << std::endl << yTrain;
    const mlpack::regression::LinearRegression linear(xTrain, yTrain);
    const arma::mat xPred = arma::randu<arma::mat>(2, 5);
    arma::rowvec yPred;
    linear.Predict(xPred, yPred);
    LOG(INFO) << "预测数据x" << std::endl << xPred;
    LOG(INFO) << "预测数据y" << std::endl << yPred;
    // 差值
    LOG(INFO) << "训练数据差值" << std::endl << rand;
    LOG(INFO) << "预测数据差值" << std::endl << yPred - arma::sum(xPred);
}
