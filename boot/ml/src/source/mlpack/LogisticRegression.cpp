#include "../../header/MLPack.hpp"
// #include <mlpack/core.hpp>
// #include <mlpack/methods/logistic_regression.hpp>

void loadFile(arma::mat& mat) {
    mlpack::data::DatasetInfo mapping;
    mlpack::data::Load("D:\\tmp\\ml\\iris.csv", mat, mapping);
    mat.shed_col(0);
    mat.shed_row(0);
    mat.row(4) -= 1;
    for(int col = mat.n_cols - 1; col >= 0; --col) {
        if(mat.col(col)[4] > 1) {
            mat.shed_col(col);
        }
    }
    LOG(INFO) << std::endl << mat.t() << std::endl;
    mat = arma::shuffle(mat, 1);
    LOG(INFO) << std::endl << mat.t() << std::endl;
    LOG(INFO) << mat.n_cols << " = " << mat.n_rows << std::endl;
}

void lifuren::testMLPackLogisticRegression() {
    arma::mat data;
    loadFile(data);
    int col = data.n_cols;
    int trainCol = col * 2 / 3;
    int predCol  = col - trainCol;
    arma::mat train  = data.head_cols(trainCol);
    arma::mat pred   = data.tail_cols(predCol);
    arma::mat xTrain = train.head_rows(4);
    arma::mat xPred  = pred.head_rows(4);
    arma::Row<size_t> yTrain = arma::conv_to<arma::Row<size_t>>::from(train.tail_rows(1));
    arma::Row<size_t> yPred  = arma::conv_to<arma::Row<size_t>>::from(pred.tail_rows(1));
    LOG(INFO) << std::endl << xTrain.t() << std::endl;
    LOG(INFO) << std::endl << yTrain.t() << std::endl;
    LOG(INFO) << std::endl << xPred.t() << std::endl;
    LOG(INFO) << std::endl << yPred.t() << std::endl;
    // arma::mat xTrain(
    //     "1 2 3;"
    //     "1 2 3"
    // );
    // arma::Row<size_t> yTrain("1 1 0");
    // LOG(INFO) << std::endl << xTrain << std::endl;
    // LOG(INFO) << std::endl << yTrain << std::endl;
    // mlpack::LogisticRegression<> logistic(xTrain, yTrain);
    ens::StandardSGD sgd(0.005, 1, 500000, 1e-10);
    // sgd.Shuffle() = true;
    // mlpack::LogisticRegression<> logistic(xTrain, yTrain, sgd, 0.001);
    // arma::rowvec sigmoids = 1 / (1 + arma::exp(0 - logistic.Parameters()[0] - logistic.Parameters().tail_cols(logistic.Parameters().n_elem - 1) * xTrain));
    // LOG(INFO) << std::endl << sigmoids << std::endl;
    ens::L_BFGS lbfgs;
    lbfgs.MinGradientNorm() = 1e-50;
    mlpack::LogisticRegression<> logistic(xTrain, yTrain, lbfgs, 0.0005);
    arma::Row<size_t> predictions;
    logistic.Classify(xPred, predictions);
    LOG(INFO) << std::endl << predictions.t() << std::endl;
    double accu = arma::accu(predictions == yPred);
    LOG(INFO) << "匹配数量：" << accu << " / " << yPred.n_cols << std::endl;
}
