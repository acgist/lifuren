/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 图片模型
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_IMAGE_MODEL_HPP
#define LFR_HEADER_CORE_IMAGE_MODEL_HPP

#include "torch/optim.h"

#include "lifuren/Model.hpp"

namespace lifuren::image {

/**
 * 编码器
 */
class Encoder : public torch::nn::Module {
    
private:
    torch::nn::Sequential layer{ nullptr };

public:
    Encoder(int in, int out) {
        torch::nn::Sequential layer;
        layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3)));
        // layer->push_back(torch::nn::BatchNorm2d(out));
        // layer->push_back(torch::nn::Dropout(0.3));
        // layer->push_back(torch::nn::ReLU());
        layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3)));
        // layer->push_back(torch::nn::BatchNorm2d(out));
        // layer->push_back(torch::nn::Dropout(0.3));
        // layer->push_back(torch::nn::ReLU());
        layer->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        this->layer = this->register_module("encoder", layer);
    }
    ~Encoder() {
        this->unregister_module("encoder");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->layer->forward(input);
    }

};

/**
 * 混合器
 */
class Muxer : public torch::nn::Module {

private:
    int batch;
    int channel;
    torch::Tensor  hidden_1{ nullptr };
    torch::Tensor  hidden_2{ nullptr };
    torch::nn::GRU muxer_1 { nullptr };
    torch::nn::GRU muxer_2 { nullptr };
    torch::nn::ConvTranspose2d conv_1{ nullptr };

public:
    Muxer(int batch, int channel, int in, int out, int num_layers = 1) : batch(batch), channel(channel) {
        this->hidden_1 = torch::zeros({num_layers, batch, out}).to(lifuren::getDevice());
        this->hidden_2 = torch::zeros({num_layers, batch, out}).to(lifuren::getDevice());
        this->muxer_1  = this->register_module("muxer_1", torch::nn::GRU(torch::nn::GRUOptions( in, out).num_layers(num_layers).batch_first(true)/*.dropout(0.1)*/));
        this->muxer_2  = this->register_module("muxer_2", torch::nn::GRU(torch::nn::GRUOptions(out, out).num_layers(num_layers).batch_first(true)/*.dropout(0.1)*/));
        this->conv_1   = this->register_module("conv_1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channel, channel, 2).stride(2)));
    }
    ~Muxer() {
        this->unregister_module("muxer_1");
        this->unregister_module("muxer_2");
        this->unregister_module("conv_2");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        input = input.flatten(2, 3);
        auto [o1, h1] = this->muxer_1->forward(input, this->hidden_1);
        auto [o2, h2] = this->muxer_2->forward(o1,    this->hidden_2);
        return this->conv_1->forward(o2.reshape({this->batch, this->channel, 80, 48}));
    }

};

/**
 * 解码器
 */
class Decoder : public torch::nn::Module {

    private:
    torch::nn::Sequential layer{ nullptr };

public:
    Decoder(int num_1) {
        int num_2 = num_1 / 2;
        torch::nn::Sequential layer;
        layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_1, num_2, 3)));
        layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_2,     3, 3)));
        this->layer = this->register_module("decoder", layer);
    }
    ~Decoder() {
        this->unregister_module("decoder");
    }

public:
    torch::Tensor forward(torch::Tensor input, torch::Tensor muxer_1, torch::Tensor muxer_2, torch::Tensor muxer_3) {
        return this->layer->forward(torch::cat({
            muxer_1,
            muxer_2,
            muxer_3
        }, 1));
    }

};

/**
 * 吴道子模型（视频风格迁移）
 */
class WudaoziModuleImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams  params;
    std::shared_ptr<Encoder> encoder_1{ nullptr };
    std::shared_ptr<Encoder> encoder_2{ nullptr };
    std::shared_ptr<Encoder> encoder_3{ nullptr };
    std::shared_ptr<Muxer>   muxer_1  { nullptr };
    std::shared_ptr<Muxer>   muxer_2  { nullptr };
    std::shared_ptr<Muxer>   muxer_3  { nullptr };
    std::shared_ptr<Decoder> decoder_1{ nullptr };

public:
    WudaoziModuleImpl(lifuren::config::ModelParams params = {});
    ~WudaoziModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(WudaoziModule);

/**
 * 吴道子模型（视频风格迁移）
 */
class WudaoziModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::Adam, lifuren::image::WudaoziModule, lifuren::dataset::SeqDatasetLoader> {

public:
    WudaoziModel(lifuren::config::ModelParams params = {});
    ~WudaoziModel();
    
public:
    void defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::image

#endif // END OF LFR_HEADER_CORE_IMAGE_MODEL_HPP
