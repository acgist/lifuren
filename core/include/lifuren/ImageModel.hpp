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
 * Conv + BatchNorm + ReLU|Sigmoid + AvgPool|MaxPool + Dropout
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
#ifndef LFR_ACTIVATION
#define LFR_ACTIVATION Sigmoid
#endif

#include "lifuren/Model.hpp"

namespace lifuren::image {

/**
 * 2D编码器
 */
class Encoder2d : public torch::nn::Module {
    
private:
    torch::nn::Sequential encoder_2d{ nullptr };

public:
    Encoder2d(int in, int out) {
        torch::nn::Sequential encoder_2d;
        encoder_2d->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).stride(1).padding(1)));
        encoder_2d->push_back(torch::nn::BatchNorm2d(out));
        encoder_2d->push_back(torch::nn::LFR_ACTIVATION());
        encoder_2d->push_back(torch::nn::Dropout(LFR_DROPOUT));
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
    torch::nn::Sequential encoder_3d{ nullptr };

public:
    Encoder3d(int channel, int num_layers = 3) {
        torch::nn::Sequential encoder_3d;
        for(int i = 1; i <= num_layers; ++i) {
            encoder_3d->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(channel, channel, 3).stride(1).padding(1)));
            encoder_3d->push_back(torch::nn::BatchNorm3d(channel));
            encoder_3d->push_back(torch::nn::LFR_ACTIVATION());
            if(i != num_layers) {
                encoder_3d->push_back(torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ 2, 2, 2 }).stride({1, 2, 2})));
            } else {
                encoder_3d->push_back(torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ 1, 2, 2 }).stride({1, 2, 2})));
            }
            encoder_3d->push_back(torch::nn::Dropout(LFR_DROPOUT));
        }
        this->encoder_3d = this->register_module("encoder_3d", encoder_3d);
    }
    ~Encoder3d() {
        this->unregister_module("encoder_3d");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->encoder_3d->forward(input);
    }

};

/**
 * 解码器
 */
class Decoder : public torch::nn::Module {

private:
    int batch;
    torch::nn::Sequential linear    { nullptr };
    torch::nn::Sequential decoder_2d{ nullptr };
    torch::nn::Sequential decoder_3d{ nullptr };

public:
    Decoder(int scale, int batch, int in, int out, bool brd = true) : batch(batch) {
        int w = LFR_IMAGE_WIDTH;
        int h = LFR_IMAGE_HEIGHT;
        int w_3d = w / scale;
        int h_3d = h / scale;
        torch::nn::Sequential linear;
        linear->push_back(torch::nn::Linear(h_3d * w_3d, h * w));
        linear->push_back(torch::nn::LFR_ACTIVATION());
        linear->push_back(torch::nn::Dropout(LFR_DROPOUT));
        this->linear = this->register_module("linear", linear);
        torch::nn::Sequential decoder_2d;
        decoder_2d->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).stride(1).padding(1)));
        if(brd) {
            decoder_2d->push_back(torch::nn::BatchNorm2d(out));
            decoder_2d->push_back(torch::nn::LFR_ACTIVATION());
            decoder_2d->push_back(torch::nn::Dropout(LFR_DROPOUT));
        }
        this->decoder_2d = this->register_module("decoder_2d", decoder_2d);
        torch::nn::Sequential decoder_3d;
        decoder_3d->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(LFR_VIDEO_QUEUE_SIZE, 1, 3).stride(1).padding(1)));
        decoder_3d->push_back(torch::nn::BatchNorm2d(1));
        decoder_3d->push_back(torch::nn::LFR_ACTIVATION());
        decoder_3d->push_back(torch::nn::Dropout(LFR_DROPOUT));
        this->decoder_3d = this->register_module("decoder_3d", decoder_3d);
    }
    ~Decoder() {
        this->unregister_module("linear");
        this->unregister_module("decoder_2d");
        this->unregister_module("decoder_3d");
    }

public:
    torch::Tensor forward(torch::Tensor input, torch::Tensor input_3d) {
        int w = LFR_IMAGE_WIDTH;
        int h = LFR_IMAGE_HEIGHT;
        return this->decoder_2d->forward(input.add(
            this->decoder_3d->forward(
                this->linear->forward(input_3d).reshape({this->batch, LFR_VIDEO_QUEUE_SIZE, h, w})
            )
        ));
    }
    torch::Tensor forward(torch::Tensor input, torch::Tensor input_2d, torch::Tensor input_3d) {
        int w = LFR_IMAGE_WIDTH;
        int h = LFR_IMAGE_HEIGHT;
        return this->decoder_2d->forward(
            torch::concat(
                {
                    input.add(
                        this->decoder_3d->forward(
                            this->linear->forward(input_3d).reshape({this->batch, LFR_VIDEO_QUEUE_SIZE, h, w})
                        )
                    ),
                    input_2d
                },
                1
            )
        );
    }

};

/**
 * 混合器
 */
class Muxer : public torch::nn::Module {

private:
    torch::Tensor  hidden{ nullptr };
    torch::nn::GRU gru   { nullptr };

public:
    Muxer(int scale, int batch, int num_layers = 3) {
        int w_3d = LFR_IMAGE_WIDTH  / scale;
        int h_3d = LFR_IMAGE_HEIGHT / scale;
        this->hidden = torch::zeros({ num_layers, batch, h_3d * w_3d }).to(LFR_DTYPE).to(lifuren::getDevice());
        this->gru    = this->register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(h_3d * w_3d, h_3d * w_3d).num_layers(num_layers).batch_first(true).dropout(num_layers == 1 ? 0.0 : LFR_DROPOUT)));
    }
    ~Muxer() {
        this->unregister_module("gru");
    }

public:
    torch::Tensor forward(torch::Tensor input_3d) {
        auto [o_h, h_h] = this->gru->forward(input_3d.flatten(2, 4), this->hidden);
        return o_h;
    }

};

/**
 * 吴道子模型（视频风格迁移）
 */
class WudaoziModuleImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams params;
    std::shared_ptr<Muxer>     muxer_1     { nullptr };
    std::shared_ptr<Encoder3d> encoder_3d_1{ nullptr };
    std::shared_ptr<Encoder2d> encoder_2d_1{ nullptr };
    std::shared_ptr<Encoder2d> encoder_2d_2{ nullptr };
    std::shared_ptr<Encoder2d> encoder_2d_3{ nullptr };
    std::shared_ptr<Decoder>   decoder_1   { nullptr };
    std::shared_ptr<Decoder>   decoder_2   { nullptr };
    std::shared_ptr<Decoder>   decoder_3   { nullptr };

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
