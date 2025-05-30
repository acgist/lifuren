#include "lifuren/ImageModel.hpp"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

lifuren::image::WudaoziModuleImpl::WudaoziModuleImpl(lifuren::config::ModelParams params) : params(params) {
    const int w_scale    = 6;
    const int h_scale    = 8;
    const int num_conv   = 3;
    const int batch_size = static_cast<int>(this->params.batch_size);
    this->muxer_1 = this->register_module("muxer_1", std::make_shared<lifuren::image::Muxer>(
        LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT,
        w_scale, h_scale,
        LFR_IMAGE_WIDTH * LFR_IMAGE_HEIGHT / h_scale / w_scale,
        LFR_IMAGE_WIDTH * LFR_IMAGE_HEIGHT / h_scale / w_scale,
        batch_size, 1, 3
    ));
    this->encoder_1 = this->register_module("encoder_1", std::make_shared<lifuren::image::Encoder>(3, num_conv));
    this->decoder_1 = this->register_module("decoder_1", std::make_shared<lifuren::image::Decoder>(3, num_conv));
}

lifuren::image::WudaoziModuleImpl::~WudaoziModuleImpl() {
    this->unregister_module("muxer_1");
    this->unregister_module("encoder_1");
    this->unregister_module("decoder_1");
}

torch::Tensor lifuren::image::WudaoziModuleImpl::forward(torch::Tensor input) {
    auto muxer_1   = this->muxer_1  ->forward(input.slice(1, 0, 1));
    auto encoder_1 = this->encoder_1->forward(input);
    auto decoder_1 = this->decoder_1->forward(encoder_1.mul(muxer_1));
    return decoder_1;
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

void lifuren::image::WudaoziModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss->forward(pred, label);
}
