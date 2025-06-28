/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * Layer
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_LAYER_HPP
#define LFR_HEADER_CORE_LAYER_HPP

#include "torch/nn.h"

namespace lifuren::nn {

/**
 * 下采样
 */
class DownsampleImpl : public torch::nn::Module {

private:
    torch::nn::Sequential downsample{ nullptr };

public:
    DownsampleImpl(int channels, int num_groups = 32) {
        this->downsample = this->register_module("downsample", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, { 3, 3 }).padding(1).bias(false)),
            torch::nn::SiLU(),
            torch::nn::GroupNorm(num_groups, channels),
            torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ 2, 2 }).stride({ 2, 2 }))
        ));
    }
    ~DownsampleImpl() {
        this->unregister_module("downsample");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->downsample->forward(input);
    }

};

TORCH_MODULE(Downsample);

/**
 * 上采样
 */
class UpsampleImpl : public torch::nn::Module {

private:
    torch::nn::Sequential upsample{ nullptr };

public:
    UpsampleImpl(int channels, int num_groups = 32) {
        this->upsample = this->register_module("upsample", torch::nn::Sequential(
            // torch::nn::ConvTranspose2d
            torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(false)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, { 3, 3 }).padding(1).bias(false)),
            torch::nn::SiLU(),
            torch::nn::GroupNorm(num_groups, channels)
        ));
    }
    ~UpsampleImpl() {
        this->unregister_module("upsample");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->upsample->forward(input);
    }

};

TORCH_MODULE(Upsample);

/**
 * 时间嵌入
 * TODO: 实现
 */
class TimeStepEmbeddingImpl : public torch::nn::Module {

};

TORCH_MODULE(TimeStepEmbedding);


/**
 * 自注意力
 */
class AttentionBlockImpl : public torch::nn::Module {

private:
    int channels;
    int num_heads;
    torch::nn::GroupNorm norm{ nullptr };
    torch::nn::Conv2d    qkv { nullptr };
    torch::nn::Conv2d    proj{ nullptr };

public:
    AttentionBlockImpl(int channels, int num_heads, int num_groups = 32) : channels(channels), num_heads(num_heads) {
        assert(channels % num_heads  == 0);
        assert(channels % num_groups == 0);
        auto norm = torch::nn::GroupNorm(num_groups, channels);
        auto qkv  = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels * 3, 1).bias(false));
        auto proj = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels,     1));
        this->norm = this->register_module("norm", norm);
        this->qkv  = this->register_module("qkv",  qkv);
        this->proj = this->register_module("proj", proj);
    }
    ~AttentionBlockImpl() {
        this->unregister_module("norm");
        this->unregister_module("qkv");
        this->unregister_module("proj");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        const int B = input.size(0);
        const int C = input.size(1);
        const int H = input.size(2);
        const int W = input.size(3);
        // [ B C H W ] -> [ B 3*C H W ]
        auto qkv = this->qkv->forward(this->norm->forward(input)).reshape({ B * this->num_heads, -1, H * W }).chunk(3, 1);
        auto q   = qkv[0];
        auto k   = qkv[1];
        auto v   = qkv[2];
        auto scale = 1.0 / std::sqrt(std::sqrt(C / this->num_heads));
        auto attn  = torch::einsum("bts,bcs->bct", { torch::einsum("bct,bcs->bts", { q * scale, k * scale }).softmax(-1), v }).reshape({ B, -1, H, W });
        auto h     = this->proj->forward(attn);
        return h + input;
    }

};

TORCH_MODULE(AttentionBlock);

/**
 * 残差网络
 */
class ResidualBlockImpl : public torch::nn::Module {

private:
    int out_c;
    torch::nn::Conv2d conv { nullptr };
    torch::nn::Linear dense{ nullptr };
    torch::nn::Sequential fn_1{ nullptr };
    torch::nn::Sequential fn_2{ nullptr };
    torch::nn::GroupNorm pre_norm { nullptr };
    torch::nn::GroupNorm post_norm{ nullptr };

public:
    ResidualBlockImpl(int channels, int out_c, int embedding_channels, int num_groups = 32) : out_c(out_c) {
        this->conv  = this->register_module("conv",  torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, out_c, { 1, 1 }).bias(false)));
        this->dense = this->register_module("dense", torch::nn::Linear(torch::nn::LinearOptions(embedding_channels, out_c).bias(false)));
        torch::nn::Sequential fn_1;
        torch::nn::Sequential fn_2;
        fn_1->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out_c, out_c, { 3, 3 }).padding(1).bias(false)));
        fn_1->push_back(torch::nn::SiLU());
        this->fn_1 = this->register_module("fn_1", fn_1);
        fn_2->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out_c, out_c, { 3, 3 }).padding(1).bias(false)));
        fn_2->push_back(torch::nn::SiLU());
        this->fn_2 = this->register_module("fn_2", fn_2);
        this->pre_norm  = this->register_module("pre_norm",  torch::nn::GroupNorm(num_groups, out_c));
        this->post_norm = this->register_module("post_norm", torch::nn::GroupNorm(num_groups, out_c));
    }
    ~ResidualBlockImpl() {
        this->unregister_module("conv");
        this->unregister_module("dense");
        this->unregister_module("fn_1");
        this->unregister_module("fn_2");
        this->unregister_module("pre_norm");
        this->unregister_module("post_norm");
    }

public:
    torch::Tensor forward(torch::Tensor input, torch::Tensor embedding) {
        torch::Tensor xi;
        if(input.size(1) == out_c) {
            xi = input.clone();
        } else {
            input = this->conv->forward(input);
            xi = input.clone();
        }
        torch::Tensor output = this->pre_norm->forward(input);
        output = this->fn_1->forward(output);
        output = output + this->dense->forward(embedding).unsqueeze(-1).unsqueeze(-1);
        output = this->post_norm->forward(output);
        output = this->fn_2->forward(output);
        return output + xi;
    }

};

TORCH_MODULE(ResidualBlock);

} // END OF lifuren::nn

#endif // END OF LFR_HEADER_CORE_LAYER_HPP
