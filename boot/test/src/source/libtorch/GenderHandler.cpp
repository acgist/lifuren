#include "../../header/LibTorch.hpp"

lifuren::GenderImpl::GenderImpl(int num_classes){
    // 卷积
    torch::nn::Sequential features;
    lifuren::layers::conv2dBatchNorm2dRelu(features, 3, 64, 3, 1, 1);
    lifuren::layers::conv2dBatchNorm2dRelu(features, 64, 64, 3, 1, 1);
    features->push_back(lifuren::layers::maxPool2d(2, 2));
    lifuren::layers::conv2dBatchNorm2dRelu(features, 64, 128, 3, 1, 1);
    lifuren::layers::conv2dBatchNorm2dRelu(features, 128, 128, 3, 1, 1);
    features->push_back(lifuren::layers::maxPool2d(2, 2));
    lifuren::layers::conv2dBatchNorm2dRelu(features, 128, 256, 3, 1, 1);
    lifuren::layers::conv2dBatchNorm2dRelu(features, 256, 256, 3, 1, 1);
    features->push_back(lifuren::layers::maxPool2d(2, 2));
    lifuren::layers::conv2dBatchNorm2dRelu(features, 256, 512, 3, 1, 1);
    lifuren::layers::conv2dBatchNorm2dRelu(features, 512, 512, 3, 1, 1);
    features->push_back(lifuren::layers::maxPool2d(2, 2));
    lifuren::layers::conv2dBatchNorm2dRelu(features, 512, 512, 3, 1, 1);
    lifuren::layers::conv2dBatchNorm2dRelu(features, 512, 512, 3, 1, 1);
    features->push_back(lifuren::layers::maxPool2d(2, 2));
    this->features = register_module("features", features);
    // 池化
    this->avgPool  = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));
    // 分类
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
    x = torch::flatten(x, 1);
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
    std::string path_val   = (data_path / "val").u8string();
    std::string path_train = (data_path / "train").u8string();
    auto data_loader_val   = lifuren::datasets::loadImageDataset(batch_size, path_val,  image_type);
    auto data_loader_train = lifuren::datasets::loadImageDataset(batch_size, path_train,image_type);
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        if (epoch == int(num_epochs / 2)) {
            learning_rate /= 10;
        }
        torch::optim::Adam optimizer(this->model->parameters(), learning_rate);
        this->trian(epoch, batch_size, optimizer, data_loader_train);
        this->val(epoch, batch_size, data_loader_val);
    }
    torch::save(this->model, save_path);
}

void lifuren::GenderHandler::trian(
    int epoch,
    int batch_size,
    torch::optim::Optimizer& optimizer,
    lifuren::datasets::ImageDatasetType& dataset
) {
    float acc_train  = 0.0;
    float loss_train = 0.0;
    size_t batch_index = 0;
    this->model->train();
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
        batch_index++;
        std::cout << "Epoch: " << epoch << " - " << batch_index << " | Train Loss: " << loss.item<float>() << " - " << loss_train / batch_index << " | Train Acc: " << acc_train / batch_index << "\r";
    }
    std::cout << std::endl;
}

void lifuren::GenderHandler::val(
    int epoch,
    int batch_size,
    lifuren::datasets::ImageDatasetType& dataset
) {
    float acc_val  = 0.0;
    float loss_val = 0.0;
    size_t batch_index = 0;
    this->model->eval();
    for (auto& batch : *dataset) {
        auto data   = batch.data.to(torch::kF32).to(device).div(255.0);
        auto target = batch.target.squeeze().to(torch::kInt64).to(device);
        torch::Tensor prediction = this->model->forward(data);
        torch::Tensor loss = torch::nll_loss(prediction, target);
        auto acc = prediction.argmax(1).eq(target).sum();
        acc_val  += acc.template item<float>() / batch_size;
        loss_val += loss.template item<float>();
        batch_index++;
        std::cout << "Epoch: " << epoch << " - " << batch_index << " | Val Loss: " << loss.item<float>() << " - " << loss_val / batch_index << " | Val Acc: " << acc_val / batch_index << "\r";
    }
    std::cout << std::endl;
}

void lifuren::GenderHandler::test(
    const std::string& data_dir,
    const std::string& image_type
) {
    // 没有测试
}

int lifuren::GenderHandler::pred(cv::Mat& image) {
    cv::resize(image, image, cv::Size(224, 224));
    torch::Tensor image_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({2, 0, 1});
    image_tensor = image_tensor.to(device).unsqueeze(0).to(torch::kF32).div(255.0);
    auto prediction = this->model->forward(image_tensor);
    prediction = torch::softmax(prediction, 1);
    auto class_id = prediction.argmax(1);
    int ans = int(class_id.item().toInt());
    float prob = prediction[0][ans].item().toFloat();
    return ans;
}
