#include "lifuren/Test.hpp"

#include <random>
#include <memory>

#include "torch/torch.h"

#include "opencv2/opencv.hpp"

#include "lifuren/Model.hpp"
#include "lifuren/Layer.hpp"
#include "lifuren/Tensor.hpp"
#include "lifuren/ImageDataset.hpp"

class GenderModuleImpl : public torch::nn::Module {

public:
    torch::nn::Sequential feature   { nullptr }; // 卷积层
    torch::nn::Sequential pool      { nullptr }; // 池化层
    torch::nn::Sequential classifier{ nullptr }; // 全连接层

public:
    GenderModuleImpl() {
        // 卷积
        torch::nn::Sequential feature;
        feature->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 4, 3)));
        feature->push_back(torch::nn::BatchNorm2d(4));
        feature->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        feature->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 8, 3)));
        feature->push_back(torch::nn::BatchNorm2d(8));
        feature->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        feature->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        this->feature = register_module("feature", feature);
        // 池化
        torch::nn::Sequential pool;
        // pool->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(32)));
        pool->push_back(torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions(32)));
        this->pool = register_module("pool", pool);
        // 分类
        torch::nn::Sequential classifier;
        classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(8 * 32 * 32, 2048)));
        classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        // classifier->push_back(torch::nn::Dropout());
        classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(2048, 512)));
        classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        // classifier->push_back(torch::nn::Dropout());
        classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(512, 128)));
        classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        // classifier->push_back(torch::nn::Dropout());
        classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 2)));
        this->classifier = register_module("classifier", classifier);
    }
    torch::Tensor forward(torch::Tensor x) {
        x = this->feature->forward(x);
        x = this->pool->forward(x);
        x = x.flatten(1);
        x = this->classifier->forward(x);
        return torch::log_softmax(x, 1);
    }
    virtual ~GenderModuleImpl() {
    }

};

TORCH_MODULE(GenderModule);

class GenderModel : public lifuren::Model<lifuren::dataset::ImageFileDatasetLoader, float, std::string, torch::nn::CrossEntropyLoss, GenderModule, torch::optim::Adam> {

public:
    GenderModel(lifuren::ModelParams params = {
        .batch_size  = 10LL,
        .epoch_count = 10LL,
        .classify    = true,
        .check_point = true,
        .check_path  = lifuren::config::CONFIG.tmp,
        .model_name  = "gender",
    }) : Model(params, [](auto tensor) {
        return tensor.squeeze().to(torch::kInt64);
    }) {
    }
    virtual ~GenderModel() {
    }

public:
    bool defineDataset() override {
        std::filesystem::path data_path = lifuren::file::join({lifuren::config::CONFIG.tmp, "gender"});
        std::string path_val   = (data_path / "val").string();
        std::string path_train = (data_path / "train").string();
        std::map<std::string, float> mapping = {
            { "man"  , 1.0F },
            { "woman", 0.0F }
        };
        this->valDataset   = std::move(lifuren::dataset::loadImageFileDataset(200, 200, this->params.batch_size, path_val,   ".jpg", mapping));
        this->trainDataset = std::move(lifuren::dataset::loadImageFileDataset(200, 200, this->params.batch_size, path_train, ".jpg", mapping));
        return true;
    }
    float eval(std::string i) {
        cv::Mat image = cv::imread(i);
        cv::resize(image, image, cv::Size(200, 200));
        torch::Tensor image_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kF32).div(255.0);
        auto prediction = this->model->forward(image_tensor);
        prediction = torch::softmax(prediction, 1);
        SPDLOG_DEBUG("预测结果：{}", prediction);
        auto class_id = prediction.argmax(1);
        int class_val = class_id.item<int>();
        SPDLOG_DEBUG("预测结果：{} - {}", class_id.item().toInt(), prediction[0][class_val].item().toFloat());
        return class_val;
    }

};

[[maybe_unused]] static void testGender() {
    GenderModel linear;
    linear.define();
    linear.trainValAndTest(true, false);
    float pred = linear.eval(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl.png"}).string());
    SPDLOG_DEBUG("当前预测：{}", pred);
    // linear.print();
    // linear.save();
}

LFR_TEST(
    testGender();
);
