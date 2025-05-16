#include "lifuren/ImageModel.hpp"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

lifuren::image::WudaoziModuleImpl::WudaoziModuleImpl(lifuren::config::ModelParams params) : params(params) {
    // 316 * 188 - 2 - 2 / 2 = 156 * 92
    // 156 *  92 - 2 - 2 / 2 =  76 * 44
    //  76 *  44 - 2 - 2 / 2 =  36 * 20
    //  36 *  20 - 2 - 2 / 2 =  16 *  8
    this->encoder_1 = this->register_module("encoder_1", std::make_shared<lifuren::image::Encoder>( 3,  16));
    this->encoder_2 = this->register_module("encoder_2", std::make_shared<lifuren::image::Encoder>(16,  32));
    this->encoder_3 = this->register_module("encoder_3", std::make_shared<lifuren::image::Encoder>(32,  64));
    this->encoder_4 = this->register_module("encoder_4", std::make_shared<lifuren::image::Encoder>(64, 128));
    // 100 * 92
    // 100 * 44
    // 100 * 20
    // 100 *  8
    this->muxer_1 = this->register_module("muxer_1", std::make_shared<lifuren::image::Muxer>(static_cast<int>(this->params.batch_size), 92, 16));
    this->muxer_2 = this->register_module("muxer_2", std::make_shared<lifuren::image::Muxer>(static_cast<int>(this->params.batch_size), 44, 16));
    this->muxer_3 = this->register_module("muxer_3", std::make_shared<lifuren::image::Muxer>(static_cast<int>(this->params.batch_size), 20, 16));
    this->muxer_4 = this->register_module("muxer_4", std::make_shared<lifuren::image::Muxer>(static_cast<int>(this->params.batch_size),  8, 16));
    // 16  *  8 * 2 + 2 + 2 =  36 *  20
    // 36  * 20 * 2 + 2 + 2 =  76 *  44
    // 76  * 44 * 2 + 2 + 2 = 156 *  92
    // 156 * 92 * 2 + 2 + 2 = 316 * 188
    this->decoder_1 = this->register_module("decoder_1", std::make_shared<lifuren::image::Decoder>(128, 64));
    this->decoder_2 = this->register_module("decoder_2", std::make_shared<lifuren::image::Decoder>( 64, 32));
    this->decoder_3 = this->register_module("decoder_3", std::make_shared<lifuren::image::Decoder>( 32, 16));
    this->decoder_4 = this->register_module("decoder_4", std::make_shared<lifuren::image::Decoder>( 16,  3));
}

lifuren::image::WudaoziModuleImpl::~WudaoziModuleImpl() {
    this->unregister_module("encoder_1");
    this->unregister_module("encoder_2");
    this->unregister_module("encoder_3");
    this->unregister_module("encoder_4");
    this->unregister_module("muxer_1");
    this->unregister_module("muxer_2");
    this->unregister_module("muxer_3");
    this->unregister_module("muxer_4");
    this->unregister_module("decoder_1");
    this->unregister_module("decoder_2");
    this->unregister_module("decoder_3");
    this->unregister_module("decoder_4");
}

torch::Tensor lifuren::image::WudaoziModuleImpl::forward(torch::Tensor input) {
    auto encoder_1 = this->encoder_1->forward(input);
    auto encoder_2 = this->encoder_2->forward(encoder_1);
    auto encoder_3 = this->encoder_3->forward(encoder_2);
    auto encoder_4 = this->encoder_4->forward(encoder_3);
    auto muxer_1   = this->muxer_1->forward(encoder_1.slice(1, 0, 1).squeeze(1));
    auto muxer_2   = this->muxer_2->forward(encoder_2.slice(1, 0, 1).squeeze(1));
    auto muxer_3   = this->muxer_3->forward(encoder_3.slice(1, 0, 1).squeeze(1));
    auto muxer_4   = this->muxer_4->forward(encoder_4.slice(1, 0, 1).squeeze(1));
    auto decoder_1 = this->decoder_1->forward(encoder_4, muxer_4).matmul(encoder_3);
    auto decoder_2 = this->decoder_2->forward(decoder_1, muxer_3).matmul(encoder_2);
    auto decoder_3 = this->decoder_3->forward(decoder_2, muxer_2).matmul(encoder_1);
    auto decoder_4 = this->decoder_4->forward(decoder_3, muxer_1).matmul(input);
    return decoder_4;
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

void lifuren::image::WudaoziModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss->forward(pred, label);
}
