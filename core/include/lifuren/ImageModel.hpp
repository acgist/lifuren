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
 * Conv + BatchNorm + ReLU|Tanh|Sigmoid + AvgPool|MaxPool + Dropout
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_IMAGE_MODEL_HPP
#define LFR_HEADER_CORE_IMAGE_MODEL_HPP

#include <cmath>

#ifndef LFR_DROPOUT
#define LFR_DROPOUT 0.3
#endif

#include "lifuren/Model.hpp"

namespace lifuren::image {

/**
 * 变形
 */
class Reshape : public torch::nn::Module {

private:
    std::vector<int64_t> shape;

public:
    Reshape(std::vector<int64_t> shape) : shape(shape) {
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return torch::reshape(input, this->shape);
    }

};

/**
 * 2D编码器
 */
class Encoder2d : public torch::nn::Module {
    
private:
    torch::nn::Sequential encoder_2d{ nullptr };

public:
    Encoder2d(int in, int out, bool output = false) {
        torch::nn::Sequential encoder_2d;
        // flatten
        encoder_2d->push_back(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1).end_dim(2)));
        // conv
        encoder_2d->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).stride(1).padding(1)));
        encoder_2d->push_back(torch::nn::BatchNorm2d(out));
        encoder_2d->push_back(torch::nn::Tanh());
        encoder_2d->push_back(torch::nn::MaxPool2d(2));
        // conv
        encoder_2d->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).stride(1).padding(1)));
        encoder_2d->push_back(torch::nn::BatchNorm2d(out));
        encoder_2d->push_back(torch::nn::Tanh());
        encoder_2d->push_back(torch::nn::MaxPool2d(2));
        // conv
        encoder_2d->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).stride(1).padding(1)));
        encoder_2d->push_back(torch::nn::MaxPool2d(2));
        this->encoder_2d = this->register_module("encoder_2d", encoder_2d);
    }
    ~Encoder2d() {
        this->unregister_module("encoder_2d");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->encoder_2d->forward(input);
    }

};

/**
 * 3D编码器
 */
class Encoder3d : public torch::nn::Module {
    
private:
    int channel;
    torch::nn::Sequential encoder_3d{ nullptr };

public:
    Encoder3d(int h_3d, int w_3d, int channel) : channel(channel) {
        torch::nn::Sequential encoder_3d;
        // conv
        encoder_3d->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(channel - 1, channel - 1, 3).stride(1).padding(1)));
        encoder_3d->push_back(torch::nn::BatchNorm3d(channel - 1));
        encoder_3d->push_back(torch::nn::Tanh());
        encoder_3d->push_back(torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ 2, 2, 2 }).stride({ 1, 2, 2 })));
        // conv
        encoder_3d->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(channel - 1, channel - 1, 3).stride(1).padding(1)));
        encoder_3d->push_back(torch::nn::BatchNorm3d(channel - 1));
        encoder_3d->push_back(torch::nn::Tanh());
        encoder_3d->push_back(torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ 2, 2, 2 }).stride({ 1, 2, 2 })));
        // conv
        encoder_3d->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(channel - 1, channel - 1, { 1, 3, 3 }).stride(1).padding({ 0, 1, 1 })));
        encoder_3d->push_back(torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ 1, 2, 2 }).stride({ 1, 2, 2 })));
        // out
        encoder_3d->push_back(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2).end_dim(4)));
        encoder_3d->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ h_3d * w_3d })));
        this->encoder_3d = this->register_module("encoder_3d", encoder_3d);

    }
    ~Encoder3d() {
        this->unregister_module("encoder_3d");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->encoder_3d->forward(input.slice(1, 1, this->channel) - input.slice(1, 0, this->channel - 1));
    }

};

/**
 * 解码器
 */
class Decoder : public torch::nn::Module {

private:
    torch::Tensor         encoder_hid{ nullptr };
    torch::nn::GRU        encoder_gru{ nullptr };
    torch::nn::Sequential decoder_3d { nullptr };

public:
    Decoder(int h, int w, int scale, int batch, int channel, int num_layers = 3) {
        int w_3d = w / scale;
        int h_3d = h / scale;
        // 3D编码器
        torch::nn::Sequential decoder_3d;
        decoder_3d->push_back(lifuren::image::Reshape({ batch, channel - 1, h_3d, w_3d }));
        decoder_3d->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(channel - 1, 1, 3).stride(1).padding(1)));
        decoder_3d->push_back(lifuren::image::Reshape({ batch, 1, h_3d * w_3d }));
        decoder_3d->push_back(torch::nn::Dropout(LFR_DROPOUT));
        decoder_3d->push_back(torch::nn::Linear(h_3d * w_3d, h * w));
        decoder_3d->push_back(lifuren::image::Reshape({ batch, 1, h, w }));
        decoder_3d->push_back(torch::nn::Tanh());
        this->decoder_3d = this->register_module("decoder_3d", decoder_3d);
        // GRU
        this->encoder_hid = torch::zeros({ num_layers, batch, h_3d * w_3d }).to(LFR_DTYPE).to(lifuren::getDevice());
        this->encoder_gru = this->register_module("encoder_gru", torch::nn::GRU(torch::nn::GRUOptions(h_3d * w_3d, h_3d * w_3d).num_layers(num_layers).batch_first(true).dropout(num_layers == 1 ? 0.0 : LFR_DROPOUT)));
    }
    ~Decoder() {
        this->unregister_module("decoder_3d");
        this->unregister_module("encoder_gru");
    }

public:
    torch::Tensor forward(torch::Tensor input_2d, torch::Tensor input_3d) {
        auto [ o_o, o_h ] = this->encoder_gru->forward(input_3d, this->encoder_hid);
        return this->decoder_3d->forward(o_o);
    }

};

/**
 * 吴道子模型（视频风格迁移）
 */
class WudaoziModuleImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams params;
    std::shared_ptr<Encoder2d> encoder_2d_1{ nullptr };
    std::shared_ptr<Encoder3d> encoder_3d_1{ nullptr };
    std::shared_ptr<Decoder>   decoder_1   { nullptr };
    
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
class WudaoziModel : public lifuren::Model<torch::optim::Adam, lifuren::image::WudaoziModule, lifuren::dataset::SeqDatasetLoader> {

public:
    WudaoziModel(lifuren::config::ModelParams params = {});
    ~WudaoziModel();
    
public:
    void defineDataset()   override;
    void defineOptimizer() override;
    torch::Tensor loss(torch::Tensor& label, torch::Tensor& pred) override;

};

} // END OF lifuren::image

#endif // END OF LFR_HEADER_CORE_IMAGE_MODEL_HPP
