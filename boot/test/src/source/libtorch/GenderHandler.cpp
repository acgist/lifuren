#include "../../header/LibTorch.hpp"

#include "FileDataset.hpp"

static auto mapping = [](const std::string& pathRef) {
    cv::Mat image = cv::imread(pathRef);
    cv::resize(image, image, cv::Size(224, 224));
    torch::Tensor data_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({2, 0, 1});
    return data_tensor;
};

lifuren::GenderImpl::GenderImpl(std::vector<int>& cfg, int num_classes, bool batch_norm){
    this->features = register_module("features", lifuren::makeFeatures(cfg, batch_norm));
    this->avgPool  = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));
    torch::nn::Sequential classifier;
    classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)));
    classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    classifier->push_back(torch::nn::Dropout());
    classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)));
    classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    classifier->push_back(torch::nn::Dropout());
    classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes)));
    this->classifier = register_module("classifier", classifier);
}

torch::Tensor lifuren::GenderImpl::forward(torch::Tensor x) {
    x = this->features->forward(x);
    x = this->avgPool(x);
    x = torch::flatten(x,1);
    x = this->classifier->forward(x);
    return torch::log_softmax(x, 1);
}

void lifuren::GenderHandler::trainAndVal(
    int   num_epochs,
    int   batch_size,
    float learning_rate,
    const std::string& data_dir,
    const std::string& image_type,
    const std::string& save_path
) {
    std::filesystem::path data_path = data_dir;
    std::string path_train = (data_path / "train").u8string();
    std::string path_val   = (data_path / "val").u8string();
    auto custom_dataset_val   = lifuren::FileDataset(path_train, { image_type }, mapping).map(torch::data::transforms::Stack<>());
    auto custom_dataset_train = lifuren::FileDataset(path_train, { image_type }, mapping).map(torch::data::transforms::Stack<>());
    auto data_loader_val   = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_val), batch_size);
    auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_train), batch_size);
    // size_t valSize   = data_loader_val.size().value();
    // size_t trainSize = data_loader_train.size().value();
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        if (epoch == int(num_epochs / 2)) {
            learning_rate /= 10;
        }
        torch::optim::Adam optimizer(this-model->parameters(), learning_rate);
        this->train(optimizer, data_loader_train);
        this->val(data_loader_val);
    }
    torch::save(this->model, save_path);
}

void lifuren::GenderHandler::train(
    torch::optim::Optimizer& optimizer,
    std::unique_ptr<torch::data::StatefulDataLoader<Dataset>> dataset
) {
    float acc_train  = 0.0;
    float loss_train = 0.0;
    size_t batch_index_train = 0;
    this->model.train();
    for (auto& batch : *dataset) {
        auto data   = batch.data.to(torch::kF32).to(this->device).div(255.0);
        auto target = batch.target.squeeze().to(torch::kInt64).to(this->device);
        optimizer.zero_grad();
        torch::Tensor prediction = this->model->forward(data);
        torch::Tensor loss = torch::nll_loss(prediction, target);
        loss.backward();
        optimizer.step();
        auto acc = prediction.argmax(1).eq(target).sum();
        acc_train  += acc.item<float>() / batch_size;
        loss_train += loss.item<float>();
        batch_index_train++;
        std::cout << "Epoch: " << epoch << " |Train Loss: " << loss_train / batch_index_train << " |Train Acc:" << acc_train / batch_index_train << "\r";
    }
    std::cout << std::endl;
}

void lifuren::GenderHandler::val(
    std::unique_ptr<torch::data::StatefulDataLoader<Dataset>> dataset
) {
    float acc_val  = 0.0;
    float loss_val = 0.0;
    size_t batch_index_val = 0;
    this->model->eval();
    for (auto& batch : *dataset) {
        auto data   = batch.data.to(torch::kF32).to(device).div(255.0);
        auto target = batch.target.squeeze().to(torch::kInt64).to(device);
        torch::Tensor prediction = this->model->forward(data);
        torch::Tensor loss = torch::nll_loss(prediction, target);
        auto acc = prediction.argmax(1).eq(target).sum();
        acc_val  += acc.template item<float>() / batch_size;
        loss_val += loss.template item<float>();
        batch_index_val++;
        std::cout << "Epoch: " << epoch << " |Val Loss: " << loss_val / batch_index_val << " |Valid Acc:" << acc_val / batch_index_val << "\r";
    }
    std::cout << std::endl;
}

void lifuren::GenderHandler::test(std::unique_ptr<torch::data::StatefulDataLoader<Dataset>> dataset) {
    // 没有测试
}

int lifuren::GenderHandler::pred(cv::Mat& image) {
    cv::resize(image, image, cv::Size(448, 448));
    torch::Tensor image_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({2, 0, 1});
    image_tensor = image_tensor.to(device).unsqueeze(0).to(torch::kF32).div(255.0);
    auto prediction = this->model->forward(image_tensor);
    prediction = torch::softmax(prediction, 1);
    auto class_id = prediction.argmax(1);
    int ans = int(class_id.item().toInt());
    float prob = prediction[0][ans].item().toFloat();
    return ans;
}
