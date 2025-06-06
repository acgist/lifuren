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

#include <cmath>

#include "lifuren/Model.hpp"

namespace lifuren::image {

/**
 * 编码器
 */
class Encoder : public torch::nn::Module {
    
private:
    torch::nn::Sequential layer{ nullptr };

public:
    Encoder(std::vector<int> layers) {
        torch::nn::Sequential layer;
        auto in  = layers.begin();
        auto out = in + 1;
        for(; out != layers.end(); ++in, ++out) {
            layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(*in, *out, 3).stride(1).padding(1)));
            if(out != layers.end() - 1) {
                layer->push_back(torch::nn::BatchNorm2d(*out));
                layer->push_back(torch::nn::ReLU());
                layer->push_back(torch::nn::Dropout(0.3));
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
    Decoder(std::vector<int> layers) {
        torch::nn::Sequential layer;
        auto in  = layers.begin();
        auto out = in + 1;
        for(; out != layers.end(); ++in, ++out) {
            layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(*in, *out, 3).stride(1).padding(1)));
            if(out != layers.end() - 1) {
                layer->push_back(torch::nn::BatchNorm2d(*out));
                layer->push_back(torch::nn::ReLU());
                layer->push_back(torch::nn::Dropout(0.3));
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
    int channel;
    torch::Tensor     hidden_h{ nullptr };
    torch::Tensor     hidden_v{ nullptr };
    torch::nn::GRU    gru_h   { nullptr };
    torch::nn::GRU    gru_v   { nullptr };
    torch::nn::Linear linear_h{ nullptr };
    torch::nn::Linear linear_v{ nullptr };

public:
    Muxer(
        int w, int h, int w_scale, int h_scale, int in, int out, int batch, int channel, int num_layers = 1
    ) : w(w), h(h), w_scale(w_scale), h_scale(h_scale), batch(batch), channel(channel) {
        this->hidden_h = torch::zeros({ num_layers, batch, out }).to(LFR_DTYPE).to(lifuren::getDevice());
        this->hidden_v = torch::zeros({ num_layers, batch, out }).to(LFR_DTYPE).to(lifuren::getDevice());
        this->gru_h    = this->register_module("gru_h", torch::nn::GRU(torch::nn::GRUOptions(in, out).num_layers(num_layers).batch_first(true).dropout(num_layers == 1 ? 0.0 : 0.3)));
        this->gru_v    = this->register_module("gru_v", torch::nn::GRU(torch::nn::GRUOptions(in, out).num_layers(num_layers).batch_first(true).dropout(num_layers == 1 ? 0.0 : 0.3)));
        this->linear_h = this->register_module("linear_h", torch::nn::Linear(in, out));
        this->linear_v = this->register_module("linear_v", torch::nn::Linear(in, out));
    }
    ~Muxer() {
        this->unregister_module("gru_h");
        this->unregister_module("gru_v");
        this->unregister_module("linear_h");
        this->unregister_module("linear_v");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        // 横向
        input = torch::layer_norm(input, {this->channel, this->h, this->w});
        auto i_h = input
            .reshape({ this->batch, this->channel * this->h_scale,                 this->h / this->h_scale, this->w                 }).permute({ 0, 1, 3, 2 })
            .reshape({ this->batch, this->channel * this->h_scale * this->w_scale, this->w / this->w_scale, this->h / this->h_scale }).permute({ 0, 1, 3, 2 });
        auto [o_h, h_h] = this->gru_h->forward(torch::relu(this->linear_h->forward(i_h.flatten(2, 3))), this->hidden_h);
        auto r_h = o_h
                                    .reshape({ this->batch, this->channel * this->h_scale * this->w_scale, this->h / this->h_scale, this->w / this->w_scale })
            .permute({ 0, 1, 3, 2 }).reshape({ this->batch, this->channel * this->h_scale,                 this->w,                 this->h / this->h_scale })
            .permute({ 0, 1, 3, 2 }).reshape({ this->batch, this->channel,                                 this->h,                 this->w                 });
        // 竖向
        auto i_v = input
            .transpose(2, 3)
            .reshape({ this->batch, this->channel * this->w_scale,                 this->w / this->w_scale, this->h                 }).permute({ 0, 1, 3, 2 })
            .reshape({ this->batch, this->channel * this->w_scale * this->h_scale, this->h / this->h_scale, this->w / this->w_scale }).permute({ 0, 1, 3, 2 })
            .transpose(2, 3);
        auto [o_v, h_v] = this->gru_v->forward(torch::relu(this->linear_v->forward(i_v.flatten(2, 3))), this->hidden_v);
        auto r_v = o_v
                                    .reshape({ this->batch, this->channel * this->h_scale * this->w_scale, this->h / this->h_scale, this->w / this->w_scale })
            .transpose(2, 3)
            .permute({ 0, 1, 3, 2 }).reshape({ this->batch, this->channel * this->w_scale,                 this->h,                 this->w / this->w_scale })
            .permute({ 0, 1, 3, 2 }).reshape({ this->batch, this->channel,                                 this->w,                 this->h                 })
            .transpose(2, 3);
        return r_h + r_v;
    }

};

/**
 * 吴道子模型（视频风格迁移）
 */
class WudaoziModuleImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams params;
    std::shared_ptr<Muxer>   muxer_1  { nullptr };
    std::shared_ptr<Encoder> encoder_1{ nullptr };
    std::shared_ptr<Decoder> decoder_1{ nullptr };

public:
    WudaoziModuleImpl(lifuren::config::ModelParams params = {});
    ~WudaoziModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);
    void          forward(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss);

};

TORCH_MODULE(WudaoziModule);

/**
 * L1Loss
 * MSELoss
 * HuberLoss
 * SmoothL1Loss
 * CrossEntropyLoss
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
