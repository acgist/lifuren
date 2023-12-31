#include "../../header/MLPack.hpp"

void loadMultFile(arma::mat& mat) {
    mlpack::data::DatasetInfo mapping;
    mlpack::data::Load("D:\\tmp\\ml\\iris.csv", mat, mapping);
    mat.shed_col(0);
    mat.shed_row(0);
    mat.row(4) -= 1;
    LOG(INFO) << std::endl << mat.t() << std::endl;
    mat = arma::shuffle(mat, 1);
    LOG(INFO) << std::endl << mat.t() << std::endl;
    LOG(INFO) << mat.n_cols << " = " << mat.n_rows << std::endl;
}

void lifuren::testMLPackSoftmaxRegression() {
    arma::mat data;
    loadMultFile(data);
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
    // mlpack::SoftmaxRegression softmax(xTrain, yTrain, 3, 0);
    // mlpack::SoftmaxRegression softmax(xTrain, yTrain, 3, 0.1);
    // mlpack::SoftmaxRegression softmax(xTrain, yTrain, 3, 0.1, true);
    // mlpack::SoftmaxRegression softmax(xTrain, yTrain, 3, 0.01);
    // mlpack::SoftmaxRegression softmax(xTrain, yTrain, 3, 0.01, true);
    // mlpack::SoftmaxRegression softmax(xTrain, yTrain, 3, 0.1, true);
    // mlpack::SoftmaxRegression softmax(xTrain, yTrain, 3, 10, true);
    // mlpack::SoftmaxRegression softmax(xTrain, yTrain, 3, 20, true);
    // arma::Row<size_t> predictions;
    // softmax.Classify(xPred, predictions);
    // LOG(INFO) << std::endl << predictions.t() << std::endl;
    // double accu = arma::accu(predictions == yPred);
    // LOG(INFO) << "匹配数量：" << accu << " / " << yPred.n_cols << std::endl;
    double lambda[] = {
        0,
        0.1,
        0.01,
        1,
        10,
        20
    };
    int size = sizeof(lambda) / sizeof(double);
    for (int i = 0; i < size; ++i) {
        double value = lambda[i];
        mlpack::SoftmaxRegression softmax(xTrain, yTrain, 3, value);
        mlpack::SoftmaxRegression softmaxFit(xTrain, yTrain, 3, value, true);
        arma::Row<size_t> predictions;
        arma::Row<size_t> predictionsFit;
        softmax.Classify(xPred, predictions);
        softmaxFit.Classify(xPred, predictionsFit);
        double accu = arma::accu(predictions == yPred);
        double accuFit = arma::accu(predictionsFit == yPred);
        LOG(INFO) << "匹配数量：" << accu << " | " << accuFit <<  " / " << yPred.n_cols << " lambda = " << value << std::endl;
    }
    
}
