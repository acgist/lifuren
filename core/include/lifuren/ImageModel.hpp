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
 * 下采样：抽象化
 */
class DownSampling : public torch::nn::Module {
    
private:
    torch::nn::Sequential layer{ nullptr };

public:
    DownSampling(int in, int out) {
        torch::nn::Sequential layer;
        layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3)));
        // layer->push_back(torch::nn::BatchNorm2d(out));
        // layer->push_back(torch::nn::Dropout(0.3));
        // layer->push_back(torch::nn::ReLU());
        layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3)));
        // layer->push_back(torch::nn::BatchNorm2d(out));
        // layer->push_back(torch::nn::Dropout(0.3));
        // layer->push_back(torch::nn::ReLU());
        this->layer = this->register_module("down-sampling", layer);
    }
    ~DownSampling() {
        this->unregister_module("down-sampling");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->layer->forward(input);
    }

};

/**
 * 时空：静态 + 动态
 */
class Live : public torch::nn::Module {

private:
    torch::nn::GRU live_1  { nullptr };
    torch::Tensor  hidden_1{ nullptr };
    torch::nn::GRU live_2  { nullptr };
    torch::Tensor  hidden_2{ nullptr };

public:
    Live(int batch, int gru_size, int num_layers = 3) {
        torch::nn::GRUOptions options_1(gru_size, gru_size);
        torch::nn::GRUOptions options_2(gru_size, gru_size);
        options_1.num_layers(num_layers).batch_first(true)/*.dropout(0.1)*/;
        options_2.num_layers(num_layers).batch_first(true)/*.dropout(0.1)*/;
        this->live_1 = this->register_module("live_1", torch::nn::GRU(options_1));
        this->live_2 = this->register_module("live_2", torch::nn::GRU(options_2));
        this->hidden_1 = torch::zeros({num_layers, batch, gru_size}).to(lifuren::getDevice());
        this->hidden_2 = torch::zeros({num_layers, batch, gru_size}).to(lifuren::getDevice());
    }
    ~Live() {
        this->unregister_module("live_1");
        this->unregister_module("live_2");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        auto [o1, h1] = this->live_1->forward(input, this->hidden_1);
        auto [o2, h2] = this->live_2->forward(o1,    this->hidden_2);
        return o2;
    }

};

/**
 * 上采样：具象化
 */
class UpSampling : public torch::nn::Module {

private:
    torch::nn::Sequential layer{ nullptr };

public:
    UpSampling(int in, int out) {
        torch::nn::Sequential layer;
        layer->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(in, in,  3)));
        layer->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(in, out, 3)));
        this->layer = this->register_module("up-sampling", layer);
    }
    ~UpSampling() {
        this->unregister_module("up-sampling");
    }

public:
    torch::Tensor forward(torch::Tensor input, torch::Tensor live) {
        return this->layer->forward(input.mul(live));
    }

};

/**
 * 吴道子模型（视频风格迁移）
 */
class WudaoziModuleImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams  params;
    std::shared_ptr<DownSampling> down_1{ nullptr };
    std::shared_ptr<DownSampling> down_2{ nullptr };
    std::shared_ptr<DownSampling> down_3{ nullptr };
    std::shared_ptr<DownSampling> down_4{ nullptr };
    std::shared_ptr<Live>         live_1{ nullptr };
    std::shared_ptr<Live>         live_2{ nullptr };
    std::shared_ptr<Live>         live_3{ nullptr };
    std::shared_ptr<Live>         live_4{ nullptr };
    std::shared_ptr<UpSampling>   up_4  { nullptr };
    std::shared_ptr<UpSampling>   up_3  { nullptr };
    std::shared_ptr<UpSampling>   up_2  { nullptr };
    std::shared_ptr<UpSampling>   up_1  { nullptr };

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
