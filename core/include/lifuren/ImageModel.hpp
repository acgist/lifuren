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
    Encoder(int in, int out, int num_layers = 1) {
        torch::nn::Sequential layer;
        for(int i = 1; i <= num_layers; ++i) {
            layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, i == num_layers ? out : in, 3).stride(1).padding(1)));
            if(i != num_layers) {
                layer->push_back(torch::nn::BatchNorm2d(in));
                layer->push_back(torch::nn::ReLU());
            }
        }
        this->layer = this->register_module("conv", layer);
    }
    ~Encoder() {
        this->unregister_module("conv");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->layer->forward(input);
    }

};

/**
 * 解码器
 */
class Decoder : public torch::nn::Module {

private:
    torch::nn::Sequential layer{ nullptr };

public:
    Decoder(int in, int out, int num_layers = 1) {
        torch::nn::Sequential layer;
        for(int i = 1; i <= num_layers; ++i) {
            layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, i != num_layers ? in : out, 3).stride(1).padding(1)));
            if(i != num_layers) {
                layer->push_back(torch::nn::BatchNorm2d(in));
                layer->push_back(torch::nn::ReLU());
            }
        }
        this->layer = this->register_module("conv", layer);
    }
    ~Decoder() {
        this->unregister_module("conv");
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
    int w_scale;
    int h_scale;
    int batch;
    int in_channel;
    int out_channel;
    torch::Tensor     hidden{ nullptr };
    torch::nn::GRU    gru   { nullptr };
    torch::nn::Conv2d conv  { nullptr };

public:
    Muxer(
        int w, int h, int w_scale, int h_scale, int in, int out, int batch, int int_channel, int out_channel, int num_layers = 1
    ) : w(w), h(h), w_scale(w_scale), h_scale(h_scale), batch(batch), in_channel(in_channel), out_channel(out_channel) {
        this->hidden = torch::zeros({ num_layers, batch, out }).to(LFR_DTYPE).to(lifuren::getDevice());
        this->gru    = this->register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(in, out).num_layers(num_layers).batch_first(true)/*.dropout(0.1)*/));
        this->conv   = this->register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, out_channel, 3).stride(1).padding(1)));
    }
    ~Muxer() {
        this->unregister_module("gru");
        this->unregister_module("conv");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        auto i_1 = input
            .reshape({ this->batch, this->in_channel * this->h_scale,                 this->h / this->h_scale, this->w                 }).permute({ 0, 1, 3, 2 })
            .reshape({ this->batch, this->in_channel * this->h_scale * this->w_scale, this->w / this->w_scale, this->h / this->h_scale }).permute({ 0, 1, 3, 2 });
        auto i_2 = this->conv->forward(i_1).flatten(2, 3);
        auto [o_1, h_1] = this->gru->forward(i_2, this->hidden);
        return o_1
            .reshape({ this->batch, this->in_channel * this->out_channel * this->h_scale * this->w_scale, this->h / this->h_scale, this->w / this->w_scale }).permute({ 0, 1, 3, 2 })
            .reshape({ this->batch, this->in_channel * this->out_channel * this->h_scale,                 this->w,                 this->h / this->h_scale }).permute({ 0, 1, 3, 2 })
            .reshape({ this->batch, this->in_channel * this->out_channel,                                 this->h,                 this->w                 });
    }

};

/**
 * 吴道子模型（视频风格迁移）
 */
class WudaoziModuleImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams  params;
    std::shared_ptr<Muxer>   muxer_1  { nullptr };
    std::shared_ptr<Encoder> encoder_1{ nullptr };
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
class WudaoziModel : public lifuren::Model<torch::nn::SmoothL1Loss, torch::optim::Adam, lifuren::image::WudaoziModule, lifuren::dataset::SeqDatasetLoader> {

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
