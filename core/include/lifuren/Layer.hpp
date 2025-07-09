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

#include <string>

#include "torch/nn.h"

#include "spdlog/spdlog.h"

namespace lifuren::nn {

/**
 * 下采样
 * 
 * AvgPool2d/MaxPool2d没有参数学习使用Conv2d代替
 */
class DownsampleImpl : public torch::nn::Module {

private:
    torch::nn::Sequential downsample{ nullptr };

public:
    DownsampleImpl(int channels, int num_groups = 32, int ratio_kernel_size = 2) {
        SPDLOG_INFO("downsample channels = {} num_groups = {} ratio_kernel_size = {}", channels, num_groups, ratio_kernel_size);
        this->downsample = this->register_module("downsample", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
            torch::nn::SiLU(),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channels)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, ratio_kernel_size).stride(ratio_kernel_size))
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
 * 
 * Upsample没有参数学习使用ConvTranspose2d代替
 */
class UpsampleImpl : public torch::nn::Module {

private:
    torch::nn::Sequential upsample{ nullptr };

public:
    UpsampleImpl(int channels, int num_groups = 32, int ratio_kernel_size = 2) {
        SPDLOG_INFO("upsample channels = {} num_groups = {} ratio_kernel_size = {}", channels, num_groups, ratio_kernel_size);
        assert(channels % num_groups == 0);
        this->upsample = this->register_module("upsample", torch::nn::Sequential(
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channels, channels, ratio_kernel_size).stride(ratio_kernel_size)),
            torch::nn::SiLU(),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channels)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1))
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
 * 姿势嵌入
 */
class PoseEmbeddingImpl : public torch::nn::Module {

private:
    torch::nn::Sequential pose_embedding{ nullptr };

public:
    PoseEmbeddingImpl(int in, int out) {
        SPDLOG_INFO("pose embedding in = {} out = {}", in, out);
        this->pose_embedding = this->register_module("pose_embedding", torch::nn::Sequential(
            torch::nn::Linear(in, out),
            torch::nn::SiLU(),
            torch::nn::Linear(out, out)
        ));
    }
    ~PoseEmbeddingImpl() {
        this->unregister_module("pose_embedding");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->pose_embedding->forward(input);
    }
    
};

TORCH_MODULE(PoseEmbedding);

/**
 * 时间嵌入
 */
class TimeEmbeddingImpl : public torch::nn::Module {

private:
    torch::nn::Sequential time_embedding{ nullptr };

public:
    TimeEmbeddingImpl(int T, int in, int out) {
        SPDLOG_INFO("time embedding T = {} in = {} out = {}", T, in, out);
        assert(in % 2 == 0);
        auto pos = torch::arange(T).to(torch::kFloat32);
        auto embedding = torch::arange(0, in, 2).to(torch::kFloat32) / in * std::log(10000);
        embedding = torch::exp(-embedding);
        embedding = pos.unsqueeze(1) * embedding.unsqueeze(0);
        embedding = torch::stack({ torch::sin(embedding), torch::cos(embedding) }, -1);
        embedding = embedding.view({ T, in });
        this->time_embedding = this->register_module("time_embedding", torch::nn::Sequential(
            torch::nn::Embedding::from_pretrained(embedding),
            torch::nn::Linear(in, out),
            torch::nn::SiLU(),
            torch::nn::Linear(out, out)
        ));
    }
    ~TimeEmbeddingImpl() {
        this->unregister_module("time_embedding");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->time_embedding->forward(input);
    }

};

TORCH_MODULE(TimeEmbedding);

/**
 * 位置嵌入
 */
using StepEmbedding = TimeEmbedding;

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
    AttentionBlockImpl(int channels, int num_heads, int embedding_dims, int num_groups = 32, float dropout = 0.3) {
        SPDLOG_INFO("attention block channels = {} num_heads = {} embedding_dims = {} num_groups = {} dropout = {:.1f}", channels, num_heads, embedding_dims, num_groups, dropout);
        assert(channels % num_heads  == 0);
        assert(channels % num_groups == 0);
        this->norm = this->register_module("norm", torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channels)));
        this->qkv  = this->register_module("qkv",  torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels * 3, 1)));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(embedding_dims, num_heads).dropout(dropout)));
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
        // const int C = input.size(1);
        const int H = input.size(2);
        const int W = input.size(3);
        auto qkv = this->qkv->forward(this->norm->forward(input)).reshape({ N, -1, H * W }).permute({ 1, 0, 2 }).chunk(3, 0);
        auto q   = qkv[0];
        auto k   = qkv[1];
        auto v   = qkv[2];
        auto [ h, w ] = attn->forward(q, k, v);
        h = h.permute({ 1, 0, 2 }).reshape({ N, -1, H, W });
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
    torch::nn::Sequential align    { nullptr };
    torch::nn::Linear     embedding{ nullptr };
    torch::nn::Sequential conv_1   { nullptr };
    torch::nn::Sequential conv_2   { nullptr };

public:
    ResidualBlockImpl(int in_channels, int out_channels, int embedding_dims, int num_groups = 32) {
        SPDLOG_INFO("residual block in_channels = {} out_channels = {} embedding_dims = {} num_groups = {}", in_channels, out_channels, embedding_dims, num_groups);
        if(in_channels == out_channels) {
            this->align = this->register_module("align", torch::nn::Sequential(
                torch::nn::Identity()
            ));
        } else {
            this->align  = this->register_module("align", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, { 1, 1 }))
            ));
        }
        this->embedding = this->register_module("embedding", torch::nn::Linear(torch::nn::LinearOptions(embedding_dims, out_channels)));
        this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out_channels)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)),
            torch::nn::SiLU()
        ));
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out_channels)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)),
            torch::nn::SiLU()
        ));
    }
    ~ResidualBlockImpl() {
        this->unregister_module("align");
        this->unregister_module("embedding");
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
    }

public:
    torch::Tensor forward(torch::Tensor input, torch::Tensor step) {
        input = this->align->forward(input);
        auto output = this->conv_1->forward(input);
        output = output + this->embedding->forward(step).unsqueeze(-1).unsqueeze(-1);
        output = this->conv_2->forward(output);
        return output + input;
    }

};

TORCH_MODULE(ResidualBlock);

/**
 * UNet
 */
class UNetImpl : public torch::nn::Module {

private:
    torch::nn::Conv2d     head          { nullptr };
    torch::nn::ModuleDict encoder_blocks{ nullptr };
    torch::nn::ModuleDict mixture_blocks{ nullptr };
    torch::nn::ModuleDict decoder_blocks{ nullptr };
    torch::nn::Sequential tail          { nullptr };

public:
    UNetImpl(
        int width, int height, int channels, int embedding_dims,
        size_t num_res = 2, int num_groups = 32, int num_heads = 8, int min_down_pixel = 4, int max_attn_pixel = 32,
        const std::vector<int>& scales = { 1, 2, 2, 4, 4 }
    ) {
        SPDLOG_INFO(
            "unet width = {} height = {} channels = {} embedding_dims = {} num_res = {} num_groups = {} num_heads = {} min_down_pixel = {} max_attn_pixel = {}",
            width, height, channels, embedding_dims, num_res, num_groups, num_heads, min_down_pixel, max_attn_pixel
        );
        int index = 0;
        int min_pixel = std::min(width, height);
        int num_skip_down = 0;
        int current_channels = embedding_dims;
        std::vector<std::tuple<int, int>> encoder_channels;
        torch::OrderedDict<std::string, std::shared_ptr<Module>> encoder_blocks;
        torch::OrderedDict<std::string, std::shared_ptr<Module>> mixture_blocks;
        torch::OrderedDict<std::string, std::shared_ptr<Module>> decoder_blocks;
        this->head = this->register_module("head", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, embedding_dims, 3).padding(1)));
        for (size_t i = 0; i < scales.size(); ++i) {
            auto scale = scales[i];
            for (size_t j = 0; j < num_res; ++j) {
                encoder_channels.emplace_back(current_channels, scale * embedding_dims);
                encoder_blocks.insert(
                    "res_" + std::to_string(i) + "_" + std::to_string(j),
                    lifuren::nn::ResidualBlock(current_channels, scale * embedding_dims, embedding_dims, num_groups).ptr()
                );
                current_channels = scale * embedding_dims;
            }
            if (min_pixel <= max_attn_pixel) {
                encoder_blocks.insert(
                    "attn_" + std::to_string(i),
                    lifuren::nn::AttentionBlock(current_channels, num_heads, width * height / std::pow(2, 2 * i), current_channels / num_heads).ptr()
                );
            }
            if (min_pixel > min_down_pixel) {
                encoder_blocks.insert(
                    "down_" + std::to_string(i),
                    lifuren::nn::Downsample(current_channels).ptr()
                );
                min_pixel = min_pixel / 2;
            } else {
                num_skip_down += 1;
            }
        }
        this->encoder_blocks = this->register_module("encoder", torch::nn::ModuleDict(encoder_blocks));
        {
            mixture_blocks.insert(
                "res_0",
                lifuren::nn::ResidualBlock(current_channels, current_channels, embedding_dims, num_groups).ptr()
            );
            mixture_blocks.insert(
                "attn_0",
                lifuren::nn::AttentionBlock(current_channels, num_heads, width * height / std::pow(2, 2 * scales.size()), current_channels / num_heads).ptr()
            );
            mixture_blocks.insert(
                "res_1",
                lifuren::nn::ResidualBlock(current_channels, current_channels, embedding_dims, num_groups).ptr()
            );
        }
        this->mixture_blocks = this->register_module("mixture", torch::nn::ModuleDict(mixture_blocks));
        std::reverse(encoder_channels.begin(), encoder_channels.end());
        for (int i = scales.size() - 1; i >= 0; --i) {
            if (index >= num_skip_down) {
                decoder_blocks.insert(
                    "up_" + std::to_string(index),
                    lifuren::nn::Upsample(current_channels).ptr()
                );
                min_pixel *= 2;
            }
            for (size_t j = 0; j < num_res; ++j) {
                auto [out_channels, in_channels] = encoder_channels[index * num_res + j];
                in_channels *= 2; // concat
                decoder_blocks.insert(
                    "res_" + std::to_string(index) + "_" + std::to_string(j),
                    lifuren::nn::ResidualBlock(in_channels, out_channels, embedding_dims, num_groups).ptr()
                );
                current_channels = out_channels;
            }
            if (min_pixel <= max_attn_pixel) {
                decoder_blocks.insert(
                    "attn_" + std::to_string(index),
                    lifuren::nn::AttentionBlock(current_channels, num_heads, width * height / std::pow(2, 2 * i), current_channels / num_heads).ptr()
                );
            }
            ++index;
        }
        this->decoder_blocks = this->register_module("decoder", torch::nn::ModuleDict(decoder_blocks));
        this->tail = this->register_module("tail", torch::nn::Sequential(
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, current_channels)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(current_channels, channels, 3).padding(1).bias(false))
        ));
    }
    ~UNetImpl() {
        this->unregister_module("head");
        this->unregister_module("encoder");
        this->unregister_module("mixture");
        this->unregister_module("decoder");
        this->unregister_module("tail");
    }

public:
    torch::Tensor forward(torch::Tensor input, torch::Tensor step) {
        std::vector<torch::Tensor> mix;
        input = this->head(input);
        mix.push_back(input);
        for (const auto& item: this->encoder_blocks->items()) {
            auto layer = item.second;
            if (typeid(*layer) == typeid(lifuren::nn::ResidualBlockImpl)) {
                input = layer->as<lifuren::nn::ResidualBlock>()->forward(input, step);
                mix.push_back(input);
            } else if (typeid(*layer) == typeid(lifuren::nn::AttentionBlockImpl)) {
                input = layer->as<lifuren::nn::AttentionBlock>()->forward(input);
            } else if (typeid(*layer) == typeid(lifuren::nn::DownsampleImpl)) {
                input = layer->as<lifuren::nn::Downsample>()->forward(input);
            } else {
                // -
            }
        }
        for (const auto& item: this->mixture_blocks->items()) {
            auto layer = item.second;
            if (typeid(*layer) == typeid(lifuren::nn::ResidualBlockImpl)) {
                input = layer->as<lifuren::nn::ResidualBlock>()->forward(input, step);
            } else if (typeid(*layer) == typeid(lifuren::nn::AttentionBlockImpl)) {
                input = layer->as<lifuren::nn::AttentionBlock>()->forward(input);
            } else {
                // -
            }
        }
        for (const auto& item: this->decoder_blocks->items()) {
            auto layer = item.second;
            if (typeid(*layer) == typeid(lifuren::nn::UpsampleImpl)) {
                input = layer->as<lifuren::nn::Upsample>()->forward(input);
            } else if (typeid(*layer) == typeid(lifuren::nn::ResidualBlockImpl)) {
                input = torch::concat({ input, mix.back() }, 1); // input = input + mix.back();
                input = layer->as<lifuren::nn::ResidualBlock>()->forward(input, step);
                mix.pop_back();
            } else if (typeid(*layer) == typeid(lifuren::nn::AttentionBlockImpl)) {
                input = layer->as<lifuren::nn::AttentionBlock>()->forward(input);
            } else {
                // -
            }
        }
        return this->tail->forward(input);
    }

};

TORCH_MODULE(UNet);

} // END OF lifuren::nn

#endif // END OF LFR_HEADER_CORE_LAYER_HPP
