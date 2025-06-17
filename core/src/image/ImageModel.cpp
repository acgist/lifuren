#include "lifuren/ImageModel.hpp"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

lifuren::image::WudaoziModuleImpl::WudaoziModuleImpl(lifuren::config::ModelParams params) : params(params) {
    const int scale = 8;
    const int batch_size = static_cast<int>(this->params.batch_size);
    this->muxer_1      = this->register_module("muxer_1",      std::make_shared<lifuren::image::Muxer>(scale, batch_size));
    this->encoder_3d_1 = this->register_module("encoder_3d_1", std::make_shared<lifuren::image::Encoder3d>(LFR_VIDEO_QUEUE_SIZE));
    this->encoder_2d_1 = this->register_module("encoder_2d_1", std::make_shared<lifuren::image::Encoder2d>( 3,  8));
    this->encoder_2d_2 = this->register_module("encoder_2d_2", std::make_shared<lifuren::image::Encoder2d>( 8, 16));
    this->encoder_2d_3 = this->register_module("encoder_2d_3", std::make_shared<lifuren::image::Encoder2d>(16, 32));
    this->decoder_1    = this->register_module("decoder_1",    std::make_shared<lifuren::image::Decoder>(scale, batch_size, 32,     16));
    this->decoder_2    = this->register_module("decoder_2",    std::make_shared<lifuren::image::Decoder>(scale, batch_size, 16 * 2,  8));
    this->decoder_3    = this->register_module("decoder_3",    std::make_shared<lifuren::image::Decoder>(scale, batch_size,  8 * 2,  3, false));
}

lifuren::image::WudaoziModuleImpl::~WudaoziModuleImpl() {
    this->unregister_module("muxer_1");
    this->unregister_module("encoder_3d_1");
    this->unregister_module("encoder_2d_1");
    this->unregister_module("encoder_2d_2");
    this->unregister_module("encoder_2d_3");
    this->unregister_module("decoder_1");
    this->unregister_module("decoder_2");
    this->unregister_module("decoder_3");
}

torch::Tensor lifuren::image::WudaoziModuleImpl::forward(torch::Tensor feature) {
    auto o_3d      = this->encoder_3d_1->forward(feature);
    auto muxer_1   = this->muxer_1  ->forward(o_3d);
    auto encoder_1 = this->encoder_2d_1->forward(feature.slice(1, LFR_VIDEO_QUEUE_SIZE - 1, LFR_VIDEO_QUEUE_SIZE).squeeze(1)); // feature.flatten(1, 2)
    auto encoder_2 = this->encoder_2d_2->forward(encoder_1);
    auto encoder_3 = this->encoder_2d_3->forward(encoder_2);
    auto decoder_1 = this->decoder_1->forward(encoder_3, muxer_1);
    auto decoder_2 = this->decoder_2->forward(encoder_2, decoder_1, muxer_1);
    auto decoder_3 = this->decoder_3->forward(encoder_1, decoder_2, muxer_1);
    return decoder_3;
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
