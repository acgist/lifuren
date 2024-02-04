#include "../../header/MLPack.hpp"

void loadMultFile(arma::mat& mat) {
    mlpack::data::DatasetInfo mapping;
    mlpack::data::Load("D:\\tmp\\ml\\iris-data.csv", mat, mapping);
    // mat.shed_col(0);
    // mat.shed_row(0);
    // mat.row(4) -= 1;
    LOG(INFO) << "全部数据" << std::endl << mat.t() << std::endl;
    mat = arma::shuffle(mat, 1);
    LOG(INFO) << "洗牌数据" << std::endl << mat.t() << std::endl;
    LOG(INFO) << "数据行列 = " << mat.n_cols << " * " << mat.n_rows << std::endl;
}

void lifuren::testMLPackSoftmaxRegression() {
    arma::mat data;
    loadMultFile(data);
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
    // LOG(INFO) << "预测结果" << std::endl << predictions.t() << std::endl;
    // double accu = arma::accu(predictions == yPred);
    // LOG(INFO) << "匹配数量 = " << accu << " / " << yPred.n_cols << std::endl;
    const double lambda[] = {
        0,
        0.1,
        0.5,
        0.01,
        0.05,
        0.001,
        0.005,
        1,
        2,
        5,
        10,
        20,
        50,
        100
    };
    const int size = sizeof(lambda) / sizeof(double);
    for (int i = 0; i < size; ++i) {
        const double value = lambda[i];
        const mlpack::SoftmaxRegression softmax(xTrain, yTrain, 3, value);
        const mlpack::SoftmaxRegression softmaxFit(xTrain, yTrain, 3, value, true);
        arma::Row<size_t> predictions;
        arma::Row<size_t> predictionsFit;
        softmax.Classify(xPred, predictions);
        softmaxFit.Classify(xPred, predictionsFit);
        const double accu = arma::accu(predictions == yPred);
        const double accuFit = arma::accu(predictionsFit == yPred);
        LOG(INFO) << "匹配数量：" << accu << " | " << accuFit <<  " / " << yPred.n_cols << " lambda = " << value << std::endl;
    }
    
}
