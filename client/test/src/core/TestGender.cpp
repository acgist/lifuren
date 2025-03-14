#include "lifuren/Test.hpp"

#include <memory>
#include <random>

#include "torch/nn.h"
#include "torch/optim.h"

#include "opencv2/opencv.hpp"

#include "lifuren/Model.hpp"
#include "lifuren/Torch.hpp"

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
        feature->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        feature->push_back(torch::nn::BatchNorm2d(4));
        feature->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 8, 3)));
        feature->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        feature->push_back(torch::nn::BatchNorm2d(8));
        feature->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 3)));
        feature->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        this->feature = register_module("feature", feature);
        // 池化
        torch::nn::Sequential pool;
        pool->push_back(torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions(16)));
        this->pool = register_module("pool", pool);
        // 分类
        torch::nn::Sequential classify;
        classify->push_back(torch::nn::Linear(torch::nn::LinearOptions(16 * 16 * 16, 2)));
        this->classify = register_module("classify", classify);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = this->feature->forward(x);
        x = this->pool->forward(x);
        x = x.flatten(1);
        x = this->classify->forward(x);
        return torch::log_sigmoid(x);
        // return torch::log_softmax(x, 1);
    }
    
    ~GenderModuleImpl() {
        unregister_module("feature");
        unregister_module("pool");
        unregister_module("classify");
    }

};

#include "torch/data.h"

TORCH_MODULE(GenderModule);

class GenderModel : public lifuren::Model<
    torch::nn::CrossEntropyLoss,
    torch::optim::Adam,
    GenderModule
> {

public:
    GenderModel(lifuren::config::ModelParams params = {
        .lr          = 0.0001F,
        .batch_size  = 10,
        .epoch_size  = 16,
        .classify    = true,
        .check_point = false,
        .model_name  = "gender",
        .model_path  = lifuren::config::CONFIG.tmp,
    }) : Model(params) {
    }

    ~GenderModel() {
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
        this->trainDataset = std::move(lifuren::dataset::image::loadClassifyDatasetLoader(200, 200, this->params.batch_size, path_train, mapping));
        this->valDataset   = std::move(lifuren::dataset::image::loadClassifyDatasetLoader(200, 200, this->params.batch_size, path_val,   mapping));
        this->testDataset  = std::move(lifuren::dataset::image::loadClassifyDatasetLoader(200, 200, this->params.batch_size, path_test,  mapping));
        return true;
    }


    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override {
        label = std::move(label.squeeze().to(torch::kInt64));
        Model::logic(feature, label, pred, loss);
    }

};

[[maybe_unused]] static void testTrain() {
    GenderModel model;
    model.define();
    model.trainValAndTest(true, true);
    model.print();
    model.save();
}

[[maybe_unused]] static void testPred() {
    GenderModel model;
    model.load();
    // model.load(lifuren::file::join({lifuren::config::CONFIG.tmp, "gender.checkpoint.16.pt"}).string());
    std::vector<std::string> paths {
        lifuren::file::join({lifuren::config::CONFIG.tmp, "xxc.png"     }).string(),
        lifuren::file::join({lifuren::config::CONFIG.tmp, "ycx.jpg"     }).string(),
        lifuren::file::join({lifuren::config::CONFIG.tmp, "girl.png"    }).string(),
        lifuren::file::join({lifuren::config::CONFIG.tmp, "ycx_cut.jpg" }).string(),
        lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_ai.png" }).string(),
        lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_lyf.png"}).string(),
        lifuren::file::join({lifuren::config::CONFIG.tmp, "woman_92.jpg"}).string()
    };
    for(const auto& path : paths) {
        auto image = cv::imread(path);
        cv::resize(image, image, cv::Size(200, 200));
        torch::Tensor image_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat32).div(255.0);
        auto prediction = model.pred(image_tensor);
        prediction = torch::softmax(prediction, 1);
        lifuren::logTensor("预测结果", prediction);
        auto class_id = prediction.argmax(1);
        int class_val = class_id.item<int>();
        SPDLOG_DEBUG("预测结果：{} - {} - {}", path, class_id.item().toInt(), prediction[0][class_val].item().toFloat());
    }
}

LFR_TEST(
    testTrain();
    // testPred();
);
