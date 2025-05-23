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
 * TODO: 补帧、超分辨率
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_IMAGE_MODEL_HPP
#define LFR_HEADER_CORE_IMAGE_MODEL_HPP

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
        layer->push_back(torch::nn::BatchNorm2d(out));
        layer->push_back(torch::nn::ReLU());
        layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3)));
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
    int w;
    int h;
    int batch;
    int channel;
    torch::Tensor  hidden_1{ nullptr };
    torch::nn::GRU gru_1   { nullptr };
    torch::nn::Linear linear_1{ nullptr };

public:
    Muxer(int w, int h, int batch, int channel, int in, int out, int num_layers = 1) : w(w), h(h), batch(batch), channel(channel) {
        this->hidden_1 = torch::zeros({num_layers, batch, out}).to(LFR_DTYPE).to(lifuren::getDevice());
        this->gru_1    = this->register_module("muxer_1", torch::nn::GRU(torch::nn::GRUOptions(in, out).num_layers(num_layers).batch_first(true)/*.dropout(0.1)*/));
        this->linear_1 = this->register_module("linear_1", torch::nn::Linear(w * h, w * h * 4));
    }
    ~Muxer() {
        this->unregister_module("muxer_1");
        this->unregister_module("linear_1");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        input = input.flatten(2, 3);
        auto [o_1, h_1] = this->gru_1->forward(input, this->hidden_1);
        return this->linear_1->forward(o_1).reshape({this->batch, this->channel, this->w * 2, this->h * 2});
    }

};

/**
 * 解码器
 */
class Decoder : public torch::nn::Module {

private:
    torch::nn::Sequential layer_1{ nullptr };

public:
    Decoder(int channel_1) {
        torch::nn::Sequential layer_1;
        layer_1->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(channel_1, channel_1, 3)));
        layer_1->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(channel_1,         3, 3)));
        this->layer_1 = this->register_module("decoder_1", layer_1);
    }
    ~Decoder() {
        this->unregister_module("decoder_1");
    }

public:
    torch::Tensor forward(torch::Tensor encoder, torch::Tensor muxer) {
        return this->layer_1->forward(muxer);
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
    std::shared_ptr<Muxer>   muxer_1  { nullptr };
    std::shared_ptr<Decoder> decoder_1{ nullptr };

public:
    WudaoziModuleImpl(lifuren::config::ModelParams params = {});
    ~WudaoziModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(WudaoziModule);

/**
 * L1Loss
 * MSELoss
 * HuberLoss
 * SmoothL1Loss
 * 
 * 吴道子模型（视频风格迁移）
 */
class WudaoziModel : public lifuren::Model<torch::nn::L1Loss, torch::optim::Adam, lifuren::image::WudaoziModule, lifuren::dataset::SeqDatasetLoader> {

public:
    WudaoziModel(lifuren::config::ModelParams params = {});
    ~WudaoziModel();
    
public:
    void defineDataset()   override;
    void defineOptimizer() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::image

#endif // END OF LFR_HEADER_CORE_IMAGE_MODEL_HPP
