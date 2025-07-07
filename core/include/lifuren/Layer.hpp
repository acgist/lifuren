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
 */
class DownsampleImpl : public torch::nn::Module {

private:
    torch::nn::Sequential downsample{ nullptr };

public:
    DownsampleImpl(int channels, int num_groups = 32, bool use_pool = true) {
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
    UpsampleImpl(int channels, int num_groups = 32, bool use_upsample = true) {
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
    torch::nn::Sequential time_embedding{ nullptr };

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
        this->time_embedding = this->register_module("time_embedding", torch::nn::Sequential(
            torch::nn::Embedding::from_pretrained(emb),
            torch::nn::Linear(in_dim, out_dim),
            torch::nn::SiLU(),
            torch::nn::Linear(out_dim, out_dim)
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

using StepEmbedding = TimeEmbedding;

/**
 * 姿势嵌入
 */
class PoseEmbeddingImpl : public torch::nn::Module {

private:
    torch::nn::Sequential pose_embedding{ nullptr };

public:
    PoseEmbeddingImpl(int in_dim, int out_dim) {
        this->pose_embedding = this->register_module("pose_embedding", torch::nn::Sequential(
            torch::nn::Linear(in_dim, out_dim),
            torch::nn::SiLU(),
            torch::nn::Linear(out_dim, out_dim)
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

/**
 * 动作向量模型
 * 
 * 这里使用图片生成比较简单，通过已有视频生成可以实现视频动作风格迁移。
 * 
 * 动作向量生成方式：
 * 1. 通过已有视频
 * 2. 通过图片生成
 * 3. 通过音频生成
 */
class PoseImpl : public torch::nn::Module {

private:
    lifuren::nn::ResidualBlock  res_1 { nullptr };
    lifuren::nn::ResidualBlock  res_2 { nullptr };
    torch::nn::Sequential     down_1{ nullptr };
    torch::nn::Sequential     down_2{ nullptr };
    lifuren::nn::AttentionBlock attn{ nullptr };
    torch::nn::Sequential pose{ nullptr };

public:
    PoseImpl(int channels, int embedding_channels, int num_groups = 8) {
        this->res_1 = this->register_module("res_1", lifuren::nn::ResidualBlock(channels,  8, embedding_channels, num_groups));
        this->res_2 = this->register_module("res_2", lifuren::nn::ResidualBlock(8,        16, embedding_channels, num_groups));
        this->down_1 = this->register_module("down_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 8, 3).padding(1)),
            torch::nn::SiLU(),
            torch::nn::GroupNorm(num_groups, 8),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 8, 4).stride(4))
        ));
        this->down_2 = this->register_module("down_2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, 3).padding(1)),
            torch::nn::SiLU(),
            torch::nn::GroupNorm(num_groups, 16),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, 8).stride(8))
        ));
        this->attn = this->register_module("attn", lifuren::nn::AttentionBlock(16, 8, 4 * 8, num_groups));
        this->pose = this->register_module("pose", torch::nn::Sequential(
            torch::nn::GroupNorm(num_groups, 16),
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
        size_t n_block = 2, int n_groups = 32, int attn_resolution = 32, const std::vector<int>& scales = { 1, 2, 2, 4, 4 }) {
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
                auto block = lifuren::nn::ResidualBlock(cur_c, scale * embedding_channels, embedding_channels, n_groups);
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
        lifuren::nn::ResidualBlock(cur_c, cur_c, embedding_channels, n_groups).ptr());
        middle_blocks.insert((std::stringstream() << "attn" << 0).str(),
                lifuren::nn::AttentionBlock(cur_c, 8, img_height * img_width / std::pow(2, 2 * scales.size()), cur_c / 8).ptr());
        middle_blocks.insert((std::stringstream() << "res" << 1).str(),
        lifuren::nn::ResidualBlock(cur_c, cur_c, embedding_channels, n_groups).ptr());
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
                lifuren::nn::ResidualBlock(in_channels, out_channels, embedding_channels, n_groups).ptr());
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
            torch::nn::GroupNorm(n_groups, cur_c),
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
