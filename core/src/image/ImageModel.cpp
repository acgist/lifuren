#include "lifuren/ImageModel.hpp"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

lifuren::image::WudaoziModuleImpl::WudaoziModuleImpl(lifuren::config::ModelParams params) : params(params) {
    // 316 * 188 - 2 - 2 / 2 = 156 * 92
    // 156 *  92 - 2 - 2 / 2 =  76 * 44
    //  76 *  44 - 2 - 2 / 2 =  36 * 20
    //  36 *  20 - 2 - 2 / 2 =  16 *  8
    this->encoder_1 = this->register_module("encoder_1", std::make_shared<lifuren::image::Encoder>( 3,  32));
    this->encoder_2 = this->register_module("encoder_2", std::make_shared<lifuren::image::Encoder>(32, 256));
    this->muxer_1   = this->register_module("muxer_1",   std::make_shared<lifuren::image::Muxer>(80, 48, static_cast<int>(this->params.batch_size), 256, 36 * 20, 160 / 2 * 96 / 2, 2));
    this->decoder_1 = this->register_module("decoder_1", std::make_shared<lifuren::image::Decoder>(256));
}

lifuren::image::WudaoziModuleImpl::~WudaoziModuleImpl() {
    this->unregister_module("encoder_1");
    this->unregister_module("encoder_2");
    this->unregister_module("muxer_1");
    this->unregister_module("decoder_1");
}

torch::Tensor lifuren::image::WudaoziModuleImpl::forward(torch::Tensor input) {
    auto encoder_1 = this->encoder_1->forward(input);
    auto encoder_2 = this->encoder_2->forward(encoder_1);
    auto muxer_1   = this->muxer_1  ->forward(encoder_2);
    return this->decoder_1->forward(encoder_1, muxer_1);
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
