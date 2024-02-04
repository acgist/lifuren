#include "../../header/MLPack.hpp"

void loadFile(arma::mat& mat) {
    mlpack::data::DatasetInfo mapping;
    mlpack::data::Load("D:\\tmp\\ml\\iris-data.csv", mat, mapping);
    // mat.shed_col(0);
    // mat.shed_row(0);
    // mat.row(4) -= 1;
    for(int col = mat.n_cols - 1; col >= 0; --col) {
        if(mat.col(col)[4] > 1) {
            mat.shed_col(col);
        }
    }
    LOG(INFO) << "原始数据" << std::endl << mat.t() << std::endl;
    mat = arma::shuffle(mat, 1);
    LOG(INFO) << "洗牌数据" << std::endl << mat.t() << std::endl;
    LOG(INFO) << "数据行列 = " << mat.n_cols << " * " << mat.n_rows << std::endl;
}

void lifuren::testMLPackLogisticRegression() {
    arma::mat data;
    loadFile(data);
    const int col = data.n_cols;
    const int trainCol = col * 2 / 3;
    const int predCol  = col - trainCol;
    const arma::mat train  = data.head_cols(trainCol);
    const arma::mat pred   = data.tail_cols(predCol);
    const arma::mat xTrain = train.head_rows(4);
    const arma::mat xPred  = pred.head_rows(4);
    const arma::Row<size_t> yTrain = arma::conv_to<arma::Row<size_t>>::from(train.tail_rows(1));
    const arma::Row<size_t> yPred  = arma::conv_to<arma::Row<size_t>>::from(pred.tail_rows(1));
    LOG(INFO) << "训练数据x" << std::endl << xTrain.t() << std::endl;
    LOG(INFO) << "训练数据y" << std::endl << yTrain.t() << std::endl;
    LOG(INFO) << "预测数据x" << std::endl << xPred.t() << std::endl;
    LOG(INFO) << "预测数据y" << std::endl << yPred.t() << std::endl;
    // ens::StandardSGD sgd(0.005, 1, 500000, 1e-10);
    // sgd.Shuffle() = true;
    // const mlpack::LogisticRegression<> logistic(xTrain, yTrain, sgd, 0.001);
    // arma::rowvec sigmoids = 1 / (1 + arma::exp(0 - logistic.Parameters()[0] - logistic.Parameters().tail_cols(logistic.Parameters().n_elem - 1) * xTrain));
    // LOG(INFO) << "sigmoids = " << std::endl << sigmoids.t() << std::endl;
    ens::L_BFGS lbfgs;
    lbfgs.MinGradientNorm() = 1e-50;
    const mlpack::LogisticRegression<> logistic(xTrain, yTrain, lbfgs, 0.0005);
    arma::Row<size_t> predictions;
    logistic.Classify(xPred, predictions);
    LOG(INFO) << "预测数据" << std::endl << predictions.t() << std::endl;
    double accu = arma::accu(predictions == yPred);
    LOG(INFO) << "匹配数量 = " << accu << " / " << yPred.n_cols << std::endl;
}
