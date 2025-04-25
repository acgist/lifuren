#include "lifuren/Test.hpp"

#include <memory>
#include <random>

#include "torch/nn.h"
#include "torch/optim.h"

#include "opencv2/opencv.hpp"

#include "lifuren/Model.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/image/ImageDataset.hpp"

class GenderModuleImpl : public torch::nn::Module {

public:
    torch::nn::Sequential feature { nullptr }; // 卷积层
    torch::nn::Sequential pool    { nullptr }; // 池化层
    torch::nn::Sequential classify{ nullptr }; // 全连接层

public:
    GenderModuleImpl() {
        // 卷积
        torch::nn::Sequential feature;
        feature->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 4, 3)));
        feature->push_back(torch::nn::BatchNorm2d(4));
        feature->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        // feature->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        feature->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 8, 3)));
        feature->push_back(torch::nn::BatchNorm2d(8));
        feature->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        feature->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        this->feature = register_module("feature", feature);
        // 池化
        torch::nn::Sequential pool;
        pool->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(32)));
        // pool->push_back(torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions(32)));
        this->pool = register_module("pool", pool);
        // 分类
        torch::nn::Sequential classify;
        classify->push_back(torch::nn::Linear(torch::nn::LinearOptions(8 * 32 * 32, 2048)));
        classify->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        // classify->push_back(torch::nn::Dropout());
        classify->push_back(torch::nn::Linear(torch::nn::LinearOptions(2048, 512)));
        classify->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        // classify->push_back(torch::nn::Dropout());
        classify->push_back(torch::nn::Linear(torch::nn::LinearOptions(512, 128)));
        classify->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        // classify->push_back(torch::nn::Dropout());
        classify->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 2)));
        this->classify = register_module("classify", classify);
    }
    torch::Tensor forward(torch::Tensor x) {
        x = this->feature->forward(x);
        x = this->pool->forward(x);
        x = x.flatten(1);
        x = this->classify->forward(x);
        return torch::log_softmax(x, 1);
    }
    virtual ~GenderModuleImpl() {
        unregister_module("feature");
        unregister_module("pool");
        unregister_module("classify");
    }

};

TORCH_MODULE(GenderModule);

class GenderModel : public lifuren::Model<
    lifuren::dataset::FileDatasetLoader,
    torch::nn::CrossEntropyLoss,
    torch::optim::Adam,
    GenderModule
> {

public:
    GenderModel(lifuren::config::ModelParams params = {
        .lr          = 0.001F,
        .batch_size  = 10,
        .epoch_count = 8,
        .classify    = true,
        .check_point = true,
        .model_name  = "gender",
        .check_path  = lifuren::config::CONFIG.tmp,
    }) : Model(params) {
    }
    virtual ~GenderModel() {
    }

public:
    bool defineDataset() override {
        std::filesystem::path data_path = lifuren::file::join({lifuren::config::CONFIG.tmp, "gender"});
        std::string path_train = (data_path / "train").string();
        std::string path_val   = (data_path / "val"  ).string();
        std::string path_test  = (data_path / "test" ).string();
        std::map<std::string, float> mapping = {
            { "man",   1.0F },
            { "woman", 0.0F }
        };
        this->trainDataset = std::move(lifuren::image::loadFileDatasetLoader(200, 200, this->params.batch_size, path_train, mapping));
        this->valDataset   = std::move(lifuren::image::loadFileDatasetLoader(200, 200, this->params.batch_size, path_val,   mapping));
        this->testDataset  = std::move(lifuren::image::loadFileDatasetLoader(200, 200, this->params.batch_size, path_test,  mapping));
        return true;
    }
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
        label = std::move(label.squeeze().to(torch::kInt64));
        Model::logic(feature, label, pred, loss);
    }

};

[[maybe_unused]] static void testGender() {
    GenderModel linear;
    linear.define();
    linear.trainValAndTest(true, true);
    // 预测
    cv::Mat image = cv::imread(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl.png"}).string());
    cv::resize(image, image, cv::Size(200, 200));
    torch::Tensor image_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat32).div(255.0);
    auto prediction = linear.pred(image_tensor);
    prediction = torch::softmax(prediction, 1);
    lifuren::logTensor("预测结果", prediction);
    auto class_id = prediction.argmax(1);
    int class_val = class_id.item<int>();
    SPDLOG_DEBUG("预测结果：{} - {}", class_id.item().toInt(), prediction[0][class_val].item().toFloat());
    // linear.print();
    // linear.save();
}

[[maybe_unused]] static void testLoadImageFileDataset() {
    auto loader = lifuren::image::loadFileDatasetLoader(
        200,
        200,
        5,
        lifuren::file::join({lifuren::config::CONFIG.tmp, "gender", "train"}).string(),
        {
            { "man"  , 1.0F },
            { "woman", 0.0F }
        }
    );
    lifuren::logTensor("图片特征", loader->begin()->data.sizes());
    lifuren::logTensor("图片标签", loader->begin()->target.sizes());
}

LFR_TEST(
    testGender();
    // testLoadImageFileDataset();
);
