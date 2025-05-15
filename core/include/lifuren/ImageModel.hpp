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
 * 下采样 抽象化 编码器
 */
class Encoder : public torch::nn::Module {
    
private:
    torch::nn::Sequential layer{ nullptr };

public:
    Encoder(int in, int out) {
        torch::nn::Sequential layer;
        layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3)));
        layer->push_back(torch::nn::BatchNorm2d(out));
        layer->push_back(torch::nn::Dropout(0.3));
        layer->push_back(torch::nn::ReLU());
        layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3)));
        layer->push_back(torch::nn::BatchNorm2d(out));
        layer->push_back(torch::nn::Dropout(0.3));
        layer->push_back(torch::nn::ReLU());
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
    torch::nn::GRU muxer_1 { nullptr };
    torch::nn::GRU muxer_2 { nullptr };
    torch::Tensor  hidden_1{ nullptr };
    torch::Tensor  hidden_2{ nullptr };

public:
    Muxer(int batch, int gru_size, int num_layers = 3) {
        torch::nn::GRUOptions options_1(gru_size, gru_size);
        torch::nn::GRUOptions options_2(gru_size, gru_size);
        options_1.num_layers(num_layers).batch_first(true)/*.dropout(0.1)*/;
        options_2.num_layers(num_layers).batch_first(true)/*.dropout(0.1)*/;
        this->muxer_1  = this->register_module("muxer_1", torch::nn::GRU(options_1));
        this->muxer_2  = this->register_module("muxer_2", torch::nn::GRU(options_2));
        this->hidden_1 = torch::zeros({num_layers, batch, gru_size}).to(lifuren::getDevice());
        this->hidden_2 = torch::zeros({num_layers, batch, gru_size}).to(lifuren::getDevice());
    }
    ~Muxer() {
        this->unregister_module("muxer_1");
        this->unregister_module("muxer_2");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        auto [o1, h1] = this->muxer_1->forward(input, this->hidden_1);
        auto [o2, h2] = this->muxer_2->forward(o1,    this->hidden_2);
        return o2;
    }

};

/**
 * 上采样 具象化 解码器
 */
class Decoder : public torch::nn::Module {

private:
    torch::nn::Sequential layer{ nullptr };

public:
    Decoder(int in, int out) {
        torch::nn::Sequential layer;
        torch::nn::ConvTranspose2dOptions options_1(in, in,  3);
        torch::nn::ConvTranspose2dOptions options_2(in, in,  3);
        torch::nn::ConvTranspose2dOptions options_3(in, out, 3);
        options_1.stride(2).kernel_size(2);
        options_2.stride(1).kernel_size(3);
        options_3.stride(1).kernel_size(3);
        layer->push_back(torch::nn::ConvTranspose2d(options_1));
        layer->push_back(torch::nn::ConvTranspose2d(options_2));
        layer->push_back(torch::nn::ConvTranspose2d(options_3));
        this->layer = this->register_module("decoder", layer);
    }
    ~Decoder() {
        this->unregister_module("decoder");
    }

public:
    torch::Tensor forward(torch::Tensor input, torch::Tensor muxer) {
        input.select(1, 0).mul_(muxer);
        input.select(1, 1).mul_(muxer);
        input.select(1, 2).mul_(muxer);
        return this->layer->forward(input);
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
    std::shared_ptr<Encoder> encoder_4{ nullptr };
    std::shared_ptr<Muxer>   muxer_1  { nullptr };
    std::shared_ptr<Muxer>   muxer_2  { nullptr };
    std::shared_ptr<Muxer>   muxer_3  { nullptr };
    std::shared_ptr<Muxer>   muxer_4  { nullptr };
    std::shared_ptr<Decoder> decoder_4{ nullptr };
    std::shared_ptr<Decoder> decoder_3{ nullptr };
    std::shared_ptr<Decoder> decoder_2{ nullptr };
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
class WudaoziModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::SGD, lifuren::image::WudaoziModule, lifuren::dataset::SeqDatasetLoader> {

public:
    WudaoziModel(lifuren::config::ModelParams params = {});
    ~WudaoziModel();
    
public:
    void defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::image

#endif // END OF LFR_HEADER_CORE_IMAGE_MODEL_HPP
