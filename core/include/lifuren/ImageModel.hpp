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
        layer->push_back(torch::nn::Dropout(0.1));
        layer->push_back(torch::nn::ReLU());
        layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3)));
        // layer->push_back(torch::nn::BatchNorm2d(out));
        layer->push_back(torch::nn::Dropout(0.1));
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
    int batch;
    int channel;
    torch::Tensor  hidden_1{ nullptr };
    torch::nn::GRU muxer_1 { nullptr };
    torch::nn::ConvTranspose2d conv_1{ nullptr };

public:
    Muxer(int batch, int channel, int in, int out, int num_layers = 1) : batch(batch), channel(channel) {
        this->hidden_1 = torch::zeros({num_layers, batch, out}).to(LFR_DTYPE).to(lifuren::getDevice());
        if(num_layers > 1) {
            auto muxer_1  = torch::nn::GRU(torch::nn::GRUOptions( in, out).num_layers(num_layers).batch_first(true));
            this->muxer_1 = this->register_module("muxer_1", muxer_1);
        } else {
            auto muxer_1  = torch::nn::GRU(torch::nn::GRUOptions( in, out).num_layers(num_layers).batch_first(true).dropout(0.1));
            this->muxer_1 = this->register_module("muxer_1", muxer_1);
        }
        auto conv_1  = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channel, channel, 2).stride(2));
        this->conv_1 = this->register_module("conv_1", conv_1);
    }
    ~Muxer() {
        this->unregister_module("muxer_1");
        this->unregister_module("conv_1");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        input = input.flatten(2, 3);
        auto [o_1, h_1] = this->muxer_1->forward(input, this->hidden_1);
        return this->conv_1->forward(o_1.reshape({this->batch, this->channel, 80, 48}));
    }

};

/**
 * 解码器
 */
class Decoder : public torch::nn::Module {

private:
    torch::nn::Sequential layer_1{ nullptr };
    torch::nn::Sequential layer_2{ nullptr };
    torch::nn::Sequential layer_3{ nullptr };

public:
    Decoder(int num_1, int num_2, int num_3) {
        torch::nn::Sequential layer_1;
        torch::nn::Sequential layer_2;
        torch::nn::Sequential layer_3;
        layer_1->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_1    , num_1 / 2, 3)));
        layer_1->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_1 / 2,         3, 3)));
        layer_2->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_2    , num_2 / 2, 3)));
        layer_2->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_2 / 2,         3, 3)));
        layer_3->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_3    , num_3 / 2, 3)));
        layer_3->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_3 / 2,         3, 3)));
        this->layer_1 = this->register_module("decoder_1", layer_1);
        this->layer_2 = this->register_module("decoder_2", layer_2);
        this->layer_3 = this->register_module("decoder_3", layer_3);
    }
    ~Decoder() {
        this->unregister_module("decoder_1");
        this->unregister_module("decoder_2");
        this->unregister_module("decoder_3");
    }

public:
    torch::Tensor forward(torch::Tensor input, torch::Tensor muxer_1, torch::Tensor muxer_2, torch::Tensor muxer_3) {
        return this->layer_1->forward(muxer_1).add(this->layer_2->forward(muxer_2)).add(this->layer_3->forward(muxer_3));
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
