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

#include "spdlog/spdlog.h"

namespace lifuren::nn {

/**
 * 下采样
 */
class DownsampleImpl : public torch::nn::Module {

private:
    torch::nn::Sequential downsample{ nullptr };

public:
    DownsampleImpl(int channels, int num_groups = 32, bool use_pool = false) {
        SPDLOG_INFO("Downsample channels = {} num_groups = {}", channels, num_groups);
        assert(channels % num_groups == 0);
        if(use_pool) {
            this->downsample = this->register_module("downsample", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
                torch::nn::SiLU(),
                torch::nn::GroupNorm(num_groups, channels),
                torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2))
            ));
        } else {
            this->downsample = this->register_module("downsample", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
                torch::nn::SiLU(),
                torch::nn::GroupNorm(num_groups, channels),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 2).stride(2))
            ));
        }
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
    UpsampleImpl(int channels, int num_groups = 32, bool use_upsample = false) {
        SPDLOG_INFO("上采样：{} - {}", channels, num_groups);
        assert(channels % num_groups == 0);
        if(use_upsample) {
            this->upsample = this->register_module("upsample", torch::nn::Sequential(
                torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(false)),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
                torch::nn::SiLU(),
                torch::nn::GroupNorm(num_groups, channels)
            ));
        } else {
            this->upsample = this->register_module("upsample", torch::nn::Sequential(
                torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channels, channels, 2).stride(2)),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
                torch::nn::SiLU(),
                torch::nn::GroupNorm(num_groups, channels)
            ));
        }
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
 * 位置嵌入
 */
class TimeEmbeddingImpl : public torch::nn::Module {

private:
    torch::nn::Sequential pos_embedding{ nullptr };

public:
    TimeEmbeddingImpl(int T, int in_dim, int out_dim) {
        SPDLOG_INFO("位置嵌入：{} - {} - {}", T, in_dim, out_dim);
        assert(in_dim % 2 == 0);
        auto pos = torch::arange(T).to(torch::kFloat32);
        auto emb = torch::arange(0, in_dim, 2).to(torch::kFloat32) / in_dim * std::log(10000);
        emb = torch::exp(-emb);
        emb = pos.unsqueeze(1) * emb.unsqueeze(0);
        emb = torch::stack({ torch::sin(emb), torch::cos(emb) }, -1);
        emb = emb.view({ T, in_dim });
        this->pos_embedding = this->register_module("pos_embedding", torch::nn::Sequential(
            torch::nn::Embedding::from_pretrained(emb),
            torch::nn::Linear(in_dim, out_dim),
            torch::nn::SiLU(),
            torch::nn::Linear(out_dim, out_dim)
        ));
    }
    ~TimeEmbeddingImpl() {
        this->unregister_module("pos_embedding");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->pos_embedding->forward(input);
    }

};

TORCH_MODULE(TimeEmbedding);

using StepEmbedding = TimeEmbedding;

// 步数嵌入
// 时间嵌入

/**
 * 自注意力
 */
class AttentionBlockImpl : public torch::nn::Module {

private:
    torch::nn::GroupNorm norm{ nullptr };
    torch::nn::Conv2d    qkv { nullptr };
    torch::nn::Conv2d    proj{ nullptr };
    torch::nn::MultiheadAttention attn{ nullptr };

public:
    AttentionBlockImpl(int channels, int num_heads, int embedding_channels, int num_groups = 32) {
        SPDLOG_INFO("自注意力：{} - {} - {} - {}", channels, num_heads, embedding_channels, num_groups);
        assert(channels % num_heads  == 0);
        assert(channels % num_groups == 0);
        this->norm = this->register_module("norm", torch::nn::GroupNorm(num_groups, channels));
        this->qkv  = this->register_module("qkv",  torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels * 3, 1)));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(embedding_channels, num_heads).dropout(0.1)));
        this->proj = this->register_module("proj", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 1)));
    }
    ~AttentionBlockImpl() {
        this->unregister_module("norm");
        this->unregister_module("qkv");
        this->unregister_module("attn");
        this->unregister_module("proj");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        const int N = input.size(0);
        const int C = input.size(1);
        const int H = input.size(2);
        const int W = input.size(3);
        // N C H W -> N C H*W -> C N H*W
        auto qkv = this->qkv->forward(this->norm->forward(input)).reshape({ N, -1, H * W }).permute({1, 0, 2}).chunk(3, 0);
        auto q   = qkv[0];
        auto k   = qkv[1];
        auto v   = qkv[2];
        auto [ h, w ] = attn->forward(q, k, v);
        // C N H*W -> N C H*W -> N C H W
        h = h.permute({1, 0, 2}).reshape({ N, -1, H, W });
        h = this->proj->forward(h);
        return h + input;
    }

};

TORCH_MODULE(AttentionBlock);

/**
 * 残差网络
 */
class ResidualBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential align{ nullptr };
    torch::nn::Linear dense{ nullptr };
    torch::nn::Sequential fn_1{ nullptr };
    torch::nn::Sequential fn_2{ nullptr };
    torch::nn::GroupNorm pre_norm { nullptr };
    torch::nn::GroupNorm post_norm{ nullptr };

public:
    ResidualBlockImpl(int channels, int out_c, int embedding_channels, int num_groups = 32) {
        SPDLOG_INFO("残差网络：{} - {} - {} - {}", channels, out_c, embedding_channels, num_groups);
        if(channels == out_c) {
            this->align = this->register_module("align", torch::nn::Sequential(torch::nn::Identity()));
        } else {
            this->align  = this->register_module("align", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, out_c, { 1, 1 }))
            ));
        }
        this->dense = this->register_module("dense", torch::nn::Linear(torch::nn::LinearOptions(embedding_channels, out_c)));
        this->fn_1 = this->register_module("fn_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_c, out_c, { 3, 3 }).padding(1)),
            torch::nn::SiLU()
        ));
        this->fn_2 = this->register_module("fn_2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_c, out_c, { 3, 3 }).padding(1)),
            torch::nn::SiLU()
        ));
        this->pre_norm  = this->register_module("pre_norm",  torch::nn::GroupNorm(num_groups, out_c));
        this->post_norm = this->register_module("post_norm", torch::nn::GroupNorm(num_groups, out_c));
    }
    ~ResidualBlockImpl() {
        this->unregister_module("align");
        this->unregister_module("dense");
        this->unregister_module("fn_1");
        this->unregister_module("fn_2");
        this->unregister_module("pre_norm");
        this->unregister_module("post_norm");
    }

public:
    torch::Tensor forward(torch::Tensor input, torch::Tensor embedding) {
        input = this->align->forward(input);
        torch::Tensor output = this->pre_norm->forward(input);
        output = this->fn_1->forward(output);
        output = output + this->dense->forward(embedding).unsqueeze(-1).unsqueeze(-1);
        output = this->post_norm->forward(output);
        output = this->fn_2->forward(output);
        return output + input;
    }

};

TORCH_MODULE(ResidualBlock);

} // END OF lifuren::nn

#endif // END OF LFR_HEADER_CORE_LAYER_HPP
