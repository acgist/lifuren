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
 * 
 * AvgPool2d/MaxPool2d没有参数学习使用Conv2d代替
 */
class DownsampleImpl : public torch::nn::Module {

private:
    torch::nn::Sequential downsample{ nullptr };

public:
    DownsampleImpl(int channels, int num_groups = 32, int ratio_kernel_size = 2) {
        SPDLOG_INFO("downsample channels = {} num_groups = {}", channels, num_groups);
        assert(channels % num_groups == 0);
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
        SPDLOG_INFO("upsample channels = {} num_groups = {}", channels, num_groups);
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
    AttentionBlockImpl(int channels, int num_heads, int embedding_channels, int num_groups = 32, float dropout = 0.3) {
        SPDLOG_INFO("attention block channels = {} num_heads = {} embedding_channels = {} num_groups = {} dropout = {:.1f}", channels, num_heads, embedding_channels, num_groups, dropout);
        assert(channels % num_heads  == 0);
        assert(channels % num_groups == 0);
        this->norm = this->register_module("norm", torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channels)));
        this->qkv  = this->register_module("qkv",  torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels * 3, 1)));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(embedding_channels, num_heads).dropout(dropout)));
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
    ResidualBlockImpl(int in_channels, int out_channels, int embedding_channels, int num_groups = 32) {
        SPDLOG_INFO("residual block in_channels = {} out_channels = {} embedding_channels = {} num_groups = {}", in_channels, out_channels, embedding_channels, num_groups);
        if(in_channels == out_channels) {
            this->align = this->register_module("align", torch::nn::Sequential(
                torch::nn::Identity()
            ));
        } else {
            this->align  = this->register_module("align", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, { 1, 1 }))
            ));
        }
        this->embedding = this->register_module("embedding", torch::nn::Linear(torch::nn::LinearOptions(embedding_channels, out_channels)));
        this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out_channels)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, { 3, 3 }).padding(1)),
            torch::nn::SiLU()
        ));
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out_channels)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, { 3, 3 }).padding(1)),
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
    torch::Tensor forward(torch::Tensor input, torch::Tensor embedding) {
        auto output = this->align->forward(input);
        output = this->conv_1->forward(output);
        output = output + this->embedding->forward(embedding).unsqueeze(-1).unsqueeze(-1);
        output = this->conv_2->forward(output);
        return output + input;
    }

};

TORCH_MODULE(ResidualBlock);

/**
 * 姿势矩阵模型
 * 
 * 这里使用图片生成比较简单，通过已有视频生成可以实现视频姿势风格迁移。
 * 
 * 姿势矩阵生成方式：
 * 1. 通过已有视频
 * 2. 通过图片生成
 * 3. 通过音频生成
 */
class PoseImpl : public torch::nn::Module {

private:
    lifuren::nn::ResidualBlock  res_1 { nullptr };
    lifuren::nn::ResidualBlock  res_2 { nullptr };
    lifuren::nn::Downsample     down_1{ nullptr };
    lifuren::nn::Downsample     down_2{ nullptr };
    lifuren::nn::AttentionBlock attn{ nullptr };
    torch::nn::Sequential pose{ nullptr };

public:
    PoseImpl(int channels, int embedding_channels, int num_groups = 8) {
        this->res_1 = this->register_module("res_1", lifuren::nn::ResidualBlock(channels,  8, embedding_channels, num_groups));
        this->res_2 = this->register_module("res_2", lifuren::nn::ResidualBlock(8,        16, embedding_channels, num_groups));
        this->down_1 = this->register_module("down_1", lifuren::nn::Downsample( 8, num_groups, 4));
        this->down_2 = this->register_module("down_2", lifuren::nn::Downsample(16, num_groups, 8));
        this->attn = this->register_module("attn", lifuren::nn::AttentionBlock(16, 8, 4 * 8, num_groups));
        this->pose = this->register_module("pose", torch::nn::Sequential(
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, 16)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 1, {3, 3}).padding(1).bias(false))
        ));
    }
    ~PoseImpl() {
    }

public:
    torch::Tensor forward(torch::Tensor input, torch::Tensor time) {
        input = this->res_1->forward(input, time);
        input = this->down_1->forward(input);
        input = this->res_2->forward(input, time);
        input = this->down_2->forward(input);
        input = this->attn->forward(input);
        return this->pose->forward(input);
    }

};

TORCH_MODULE(Pose);

/**
 * UNet
 */
class UNetImpl : public torch::nn::Module {

private:
    torch::nn::Conv2d head{ nullptr };
    torch::nn::ModuleDict encoder_blocks{nullptr};
    torch::nn::ModuleDict middle_blocks{nullptr};
    torch::nn::ModuleDict decoder_blocks{nullptr};
    torch::nn::Sequential tail{ nullptr };

public:
    UNetImpl(int img_height, int img_width, int channels, int embedding_channels,  int min_pixel = 4,
        size_t n_block = 2, int num_groups = 32, int attn_resolution = 32, const std::vector<int>& scales = { 1, 2, 2, 4, 4 }) {
        this->head = this->register_module("head", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, embedding_channels, { 3, 3 }).padding(1)));
        int min_img_size = std::min(img_height, img_width);
        torch::OrderedDict<std::string, std::shared_ptr<Module>> encoder_blocks;
        std::vector<std::tuple<int, int>> encoder_channels;
        int cur_c = embedding_channels;
        size_t skip_pooling = 0;
        for (size_t i = 0; i < scales.size(); i++) {
            auto scale = scales[i];
            for (size_t j = 0; j < n_block; j++) {
                encoder_channels.emplace_back(cur_c, scale * embedding_channels);
                auto block = lifuren::nn::ResidualBlock(cur_c, scale * embedding_channels, embedding_channels, num_groups);
                cur_c = scale * embedding_channels;
                encoder_blocks.insert((std::stringstream() << "res" << i * n_block + j).str(), block.ptr());
            }
            if (min_img_size <= attn_resolution) {
                encoder_blocks.insert((std::stringstream() << "attn" << i * n_block).str(),
                lifuren::nn::AttentionBlock(cur_c, 8, img_height * img_width / std::pow(2, 2 * i), cur_c / 8).ptr());
            }
            if (min_img_size > min_pixel) {
                encoder_blocks.insert((std::stringstream() << "down" << i).str(), lifuren::nn::Downsample(cur_c).ptr());
                min_img_size = min_img_size / 2;
            } else {
                skip_pooling += 1;
            }
        }
        this->encoder_blocks = this->register_module("encoder", torch::nn::ModuleDict(encoder_blocks));

        torch::OrderedDict<std::string, std::shared_ptr<Module>> middle_blocks;
        middle_blocks.insert((std::stringstream() << "res" << 0).str(),
        lifuren::nn::ResidualBlock(cur_c, cur_c, embedding_channels, num_groups).ptr());
        middle_blocks.insert((std::stringstream() << "attn" << 0).str(),
                lifuren::nn::AttentionBlock(cur_c, 8, img_height * img_width / std::pow(2, 2 * scales.size()), cur_c / 8).ptr());
        middle_blocks.insert((std::stringstream() << "res" << 1).str(),
        lifuren::nn::ResidualBlock(cur_c, cur_c, embedding_channels, num_groups).ptr());
        this->middle_blocks = this->register_module("muxer", torch::nn::ModuleDict(middle_blocks));

        std::reverse(encoder_channels.begin(), encoder_channels.end());

        torch::OrderedDict<std::string, std::shared_ptr<Module>> decoder_blocks;
        size_t m = 0;
        for (int i = scales.size() - 1; i > -1; i--) {
            auto rev_scale = scales[i];
            if (m >= skip_pooling) {
                decoder_blocks.insert((std::stringstream() << "up" << m).str(), lifuren::nn::Upsample(cur_c).ptr());
                min_img_size *= 2;
            }

            for (size_t j = 0; j < n_block; j++) {
                auto [out_channels, in_channels] = encoder_channels[m * n_block + j];
                in_channels *= 2;
                decoder_blocks.insert((std::stringstream() << "res" << m * n_block + j).str(),
                lifuren::nn::ResidualBlock(in_channels, out_channels, embedding_channels, num_groups).ptr());
                cur_c = out_channels;
            }

            if (min_img_size <= attn_resolution) {
                decoder_blocks.insert((std::stringstream() << "attn" << m * n_block).str(),
                lifuren::nn::AttentionBlock(cur_c, 8, img_height * img_width / std::pow(2, 2 * i), cur_c / 8).ptr());
            }

            m++;
        }
        this->decoder_blocks = this->register_module("decoder", torch::nn::ModuleDict(decoder_blocks));
        torch::nn::Sequential tail(
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, cur_c)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(cur_c, channels, {3, 3}).padding(1).bias(false))
        );
        this->tail = this->register_module("tail", tail);
    }
    ~UNetImpl() {
        this->unregister_module("head");
        this->unregister_module("tail");
        this->unregister_module("muxer");
        this->unregister_module("encoder");
        this->unregister_module("decoder");
    }

public:
    torch::Tensor forward(torch::Tensor x, torch::Tensor t) {
        x = head(x);

        std::vector<torch::Tensor> inners;
    
        inners.push_back(x);
        for (const auto &item: encoder_blocks->items()) {
            auto name = item.first;
            auto module = item.second;
            if (name.starts_with("res")) {
                x = module->as<lifuren::nn::ResidualBlock>()->forward(x, t);
                inners.push_back(x);
            } else if (name.starts_with("attn")) {
                x = module->as<lifuren::nn::AttentionBlock>()->forward(x);
            } else if (name.starts_with("down")) {
                x = module->as<lifuren::nn::Downsample>()->forward(x);
            } else {
                // -
            }
        }
    
        for (const auto &item: middle_blocks->items()) {
            auto name = item.first;
            auto module = item.second;
            if (name.starts_with("res")) {
                x = module->as<lifuren::nn::ResidualBlock>()->forward(x, t);
            } else if (name.starts_with("attn")) {
                x = module->as<lifuren::nn::AttentionBlock>()->forward(x);
            } else {
                // -
            }
        }

        auto inners_ = std::vector<torch::Tensor>(inners.begin(), inners.end());
    
        for (const auto &item: decoder_blocks->items()) {
            auto name = item.first;
            auto module = item.second;
    
            if (name.starts_with("up")) {
                x = module->as<lifuren::nn::Upsample>()->forward(x);
                torch::Tensor xi = inners_.back();
            } else if (name.starts_with("res")) {
                torch::Tensor xi = inners_.back();
                inners_.pop_back();
                // x = x + xi;
                x = torch::concat({ x, xi }, 1);
                x = module->as<lifuren::nn::ResidualBlock>()->forward(x, t);
            } else if (name.starts_with("attn")) {
                x = module->as<lifuren::nn::AttentionBlock>()->forward(x);
            } else {
                // -
            }
        }
        
        return this->tail->forward(x);
    }

};

TORCH_MODULE(UNet);

} // END OF lifuren::nn

#endif // END OF LFR_HEADER_CORE_LAYER_HPP
