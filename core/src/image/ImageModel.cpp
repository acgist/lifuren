#include "lifuren/ImageModel.hpp"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

lifuren::image::WudaoziModuleImpl::WudaoziModuleImpl(lifuren::config::ModelParams params) : params(params) {
    const int scale = 8;
    const int batch_size = static_cast<int>(this->params.batch_size);
    this->encoder_2d_1 = this->register_module("encoder_2d_1", std::make_shared<lifuren::image::Encoder2d>(3 * LFR_VIDEO_QUEUE_SIZE, 16));
    this->encoder_3d_1 = this->register_module("encoder_3d_1", std::make_shared<lifuren::image::Encoder3d>(LFR_IMAGE_HEIGHT / scale, LFR_IMAGE_WIDTH / scale, LFR_VIDEO_QUEUE_SIZE));
    this->decoder_1    = this->register_module("decoder_1",    std::make_shared<lifuren::image::Decoder>(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, scale, batch_size, LFR_VIDEO_QUEUE_SIZE));
}

lifuren::image::WudaoziModuleImpl::~WudaoziModuleImpl() {
    this->unregister_module("encoder_2d_1");
    this->unregister_module("encoder_3d_1");
    this->unregister_module("decoder_1");
}

torch::Tensor lifuren::image::WudaoziModuleImpl::forward(torch::Tensor feature) {
    auto encoder_2d_1 = this->encoder_2d_1->forward(feature);
    auto encoder_3d_1 = this->encoder_3d_1->forward(feature);
    auto decoder_1    = this->decoder_1   ->forward(encoder_2d_1, encoder_3d_1);
    return feature.slice(1, LFR_VIDEO_QUEUE_SIZE - 1, LFR_VIDEO_QUEUE_SIZE).squeeze(1).add(decoder_1);
}

lifuren::image::WudaoziModel::WudaoziModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::image::WudaoziModel::~WudaoziModel() {
}

void lifuren::image::WudaoziModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.test_path);
    }
}

void lifuren::image::WudaoziModel::defineOptimizer() {
    torch::optim::AdamOptions optims;
    optims.lr(this->params.lr);
    optims.eps(0.0001);
    this->optimizer = std::make_unique<torch::optim::Adam>(this->model->parameters(), optims);
}

torch::Tensor lifuren::image::WudaoziModel::loss(torch::Tensor& label, torch::Tensor& pred) {
    // L1Loss
    // MSELoss
    // HuberLoss
    // SmoothL1Loss
    // CrossEntropyLoss
    // return torch::smooth_l1_loss(pred, label);
    return torch::sum((pred - label).abs(), { 1, 2, 3 }, true).mean();
    // return torch::sum((pred - label).pow(2), { 1, 2, 3 }, true).mean();
}
