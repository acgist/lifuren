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

#include "lifuren/File.hpp"
#include "lifuren/Model.hpp"

#include "opencv2/opencv.hpp"

#ifndef LFR_DROPOUT
#define LFR_DROPOUT 0.3
#endif

namespace lifuren::image {

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
 * 残差网络
 */
class ResidualBlockImpl : public torch::nn::Module {

private:
    torch::nn::Conv2d conv { nullptr };
    torch::nn::Linear dense{ nullptr };
    torch::nn::Sequential fn_1{ nullptr };
    torch::nn::Sequential fn_2{ nullptr };
    torch::nn::GroupNorm pre_norm { nullptr };
    torch::nn::GroupNorm post_norm{ nullptr };

public:
    ResidualBlockImpl(int channels, int out_c, int embedding_channels, int num_groups = 32) {
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
 * UNet
 */
class UNetImpl : public torch::nn::Module {

private:
    torch::nn::Conv2d stem{ nullptr };
    torch::nn::Sequential time_embed{ nullptr };
    torch::nn::ModuleDict encoder_blocks{nullptr};
    torch::nn::ModuleDict decoder_blocks{nullptr};

public:
    UNetImpl(int img_height, int img_width, int channels, int model_channels, int embedding_channels,  int min_pixel = 4,
        int n_block = 2, int n_groups = 32, int attn_resolution = 16, const std::vector<int>& scales = { 1, 1, 2, 2, 4, 4 }) {
        this->stem = this->register_module("stem", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, embedding_channels, { 3, 3 }).padding(1)));
        this->time_embed = this->register_module("time_embed", torch::nn::Sequential(
            torch::nn::Linear(model_channels, embedding_channels),
            torch::nn::SiLU(),
            torch::nn::Linear(embedding_channels, embedding_channels)
        ));
        int min_img_size = std::min(img_height, img_width);
        torch::OrderedDict<std::string, std::shared_ptr<Module>> encoder_blocks;
        std::vector<std::tuple<int, int>> encoder_channels;
        int cur_c = embedding_channels;
        auto skip_pooling = 0;
        for (size_t i = 0; i < scales.size(); i++) {
            auto scale = scales[i];
            // sevaral residual blocks
            for (size_t j = 0; j < n_block; j++) {
                encoder_channels.emplace_back(cur_c, scale * embedding_channels);
                auto block = ResidualBlock(cur_c, scale * embedding_channels, embedding_channels, n_groups);
                cur_c = scale * embedding_channels;
                encoder_blocks.insert((std::stringstream() << "enc_block_" << i * n_block + j).str(), block.ptr());
            }
    
            if (min_img_size <= attn_resolution) {
                encoder_blocks.insert((std::stringstream() << "attn_enc_block_" << i * n_block).str(),
                                  AttentionBlock(cur_c, 8, cur_c / 8).ptr());
            }
    
            // downsample block if not reach to `min_pixel`.
            if (min_img_size > min_pixel) {
                encoder_blocks.insert((std::stringstream() << "down_block_" << i).str(), Downsample(cur_c).ptr());
                min_img_size = min_img_size / 2;
            } else {
                skip_pooling += 1; // log how many times skip pooling.
            }
        }
        // mid
        encoder_blocks.insert((std::stringstream() << "enc_block_" << scales.size() * n_block).str(),
            ResidualBlock(cur_c, cur_c, embedding_channels, n_groups).ptr());
            // ?
            // ResidualBlock(ch, ch, time_embed_dim, dropout),
            // AttentionBlock(ch, num_heads=num_heads),
            // ResidualBlock(ch, ch, time_embed_dim, dropout)

        this->encoder_blocks = this->register_module("encoder_blocks", torch::nn::ModuleDict(encoder_blocks));

        std::reverse(encoder_channels.begin(), encoder_channels.end());

        torch::OrderedDict<std::string, std::shared_ptr<Module>> decoder_blocks;
        size_t m = 0;
        for (int i = scales.size() - 1; i > -1; i--) {
            auto rev_scale = scales[i];
            if (m >= skip_pooling) {
                decoder_blocks.insert((std::stringstream() << "up_block_" << m).str(), Upsample(cur_c).ptr());
                min_img_size *= 2;
            }

            for (size_t j = 0; j < n_block; j++) {
                int out_channels;
                int in_channels;
                std::tie(out_channels, in_channels) = encoder_channels[m * n_block + j];
                decoder_blocks.insert((std::stringstream() << "dec_block_" << m * n_block + j).str(),
                                ResidualBlock(in_channels, out_channels, embedding_channels, n_groups).ptr());
                cur_c = out_channels;
            }

            if (min_img_size <= attn_resolution) {
                decoder_blocks.insert((std::stringstream() << "attn_dec_block_" << m * n_block).str(),
                                AttentionBlock(cur_c, 8, cur_c / 8).ptr());
            }

            m++;
        }
        torch::nn::Sequential out(
            torch::nn::GroupNorm(n_groups, cur_c),
                torch::nn::SiLU(),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(cur_c, channels, {3, 3}).padding(1).bias(false))
        );
        decoder_blocks.insert(std::string("out"), out.ptr());
        this->decoder_blocks = this->register_module("decoder_blocks", torch::nn::ModuleDict(decoder_blocks));
    }
    ~UNetImpl() {
        this->unregister_module("stem");
        this->unregister_module("time_embed");
    }

public:
    torch::Tensor forward(torch::Tensor x, torch::Tensor t) {
        x = stem(x);

        std::vector<torch::Tensor> inners;
    
        inners.push_back(x);
        for (const auto &item: encoder_blocks->items()) {
            auto name = item.first;
            auto module = item.second;
            // resudial block
            if (name.starts_with("enc")) {
                x = module->as<ResidualBlock>()->forward(x, t);
                inners.push_back(x);
            } else if (name.starts_with("attn")) {
                x = module->as<AttentionBlock>()->forward(x);
            }
                // downsample block
            else {
                x = module->as<torch::nn::Sequential>()->forward(x);
                inners.push_back(x);
            }
        }
    
        // drop last two (contains middle block output)
        auto inners_ = std::vector<torch::Tensor>(inners.begin(), inners.end() - 2);
    
        for (const auto &item: decoder_blocks->items()) {
            auto name = item.first;
            auto module = item.second;
    
            // upsample block
            if (name.starts_with("up")) {
                x = module->as<torch::nn::Sequential>()->forward(x);
                torch::Tensor xi = inners_.back();
                inners_.pop_back(); // pop()
                x = x + xi;
            }
                // resudial block
            else if (name.starts_with("dec")) {
                torch::Tensor xi = inners_.back();
                inners_.pop_back(); // pop()
                x = module->as<ResidualBlock>()->forward(x, t);
                x = x + xi;
            } else if (name.starts_with("attn")) {
    
                x = module->as<AttentionBlock>()->forward(x);
            } else {
                x = module->as<torch::nn::Sequential>()->forward(x);
            }
        }
    
        return x;
    }

};

TORCH_MODULE(UNet);

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
class Encoder2dImpl : public torch::nn::Module {
    
private:
    torch::nn::Sequential encoder_2d{ nullptr };

public:
    Encoder2dImpl(int in, int out, bool output = false) {
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
    ~Encoder2dImpl() {
        this->unregister_module("encoder_2d");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->encoder_2d->forward(input);
    }

};

TORCH_MODULE(Encoder2d);

/**
 * 3D编码器
 */
class Encoder3dImpl : public torch::nn::Module {
    
private:
    int channel;
    torch::nn::Sequential encoder_3d{ nullptr };

public:
    Encoder3dImpl(int h_3d, int w_3d, int channel) : channel(channel) {
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
    ~Encoder3dImpl() {
        this->unregister_module("encoder_3d");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->encoder_3d->forward(input.slice(1, 1, this->channel) - input.slice(1, 0, this->channel - 1));
    }

};

TORCH_MODULE(Encoder3d);

/**
 * 解码器
 */
class DecoderImpl : public torch::nn::Module {

private:
    torch::Tensor         encoder_hid{ nullptr };
    torch::nn::GRU        encoder_gru{ nullptr };
    torch::nn::Sequential decoder_3d { nullptr };

public:
    DecoderImpl(int h, int w, int scale, int batch, int channel, int num_layers = 3) {
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
    ~DecoderImpl() {
        this->unregister_module("decoder_3d");
        this->unregister_module("encoder_gru");
    }

public:
    torch::Tensor forward(torch::Tensor input_2d, torch::Tensor input_3d) {
        auto [ o_o, o_h ] = this->encoder_gru->forward(input_3d, this->encoder_hid);
        return this->decoder_3d->forward(o_o);
    }

};

TORCH_MODULE(Decoder);


inline torch::Tensor linear_beta_schedule(int timesteps) {
    auto scale = 1000.0 / timesteps;
    auto beta_start = scale * 0.0001;
    auto beta_end = scale * 0.02;
    return torch::linspace(beta_start, beta_end, timesteps);
}

inline torch::Tensor cosine_beta_schedule(int timesteps,  double s=0.008) {
    // auto steps = timesteps + 1;
    // auto x = torch::linspace(0, timesteps, steps);
    // auto alphas_cumprod = torch::pow(torch::cos(((x / timesteps) + s) / (1 + s) * M_PI * 0.5), 2);
    // alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    // auto betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    // return torch::clip(betas, 0, 0.999);
    return {};
}

/**
 * 吴道子模型（视频生成）
 */
class WudaoziModuleImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams params;
    Encoder2d encoder_2d_1{ nullptr };
    Encoder3d encoder_3d_1{ nullptr };
    Decoder   decoder_1   { nullptr };
    torch::Tensor betas { nullptr };
    torch::Tensor alphas { nullptr };
    
public:
    WudaoziModuleImpl(lifuren::config::ModelParams params = {}) : params(params) {
        const int scale = 8;
        const int batch_size = static_cast<int>(this->params.batch_size);
        this->encoder_2d_1 = this->register_module("encoder_2d_1", lifuren::image::Encoder2d(3 * LFR_VIDEO_QUEUE_SIZE, 16));
        this->encoder_3d_1 = this->register_module("encoder_3d_1", lifuren::image::Encoder3d(LFR_IMAGE_HEIGHT / scale, LFR_IMAGE_WIDTH / scale, LFR_VIDEO_QUEUE_SIZE));
        this->decoder_1    = this->register_module("decoder_1",    lifuren::image::Decoder(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, scale, batch_size, LFR_VIDEO_QUEUE_SIZE));
        int timesteps = 1000;
        this->betas = linear_beta_schedule(timesteps);
        this->alphas = 1.0 - this->betas;
    }
    ~WudaoziModuleImpl() {
        this->unregister_module("encoder_2d_1");
        this->unregister_module("encoder_3d_1");
        this->unregister_module("decoder_1");
    }

public:
    torch::Tensor forward(torch::Tensor feature) {
        auto encoder_2d_1 = this->encoder_2d_1->forward(feature);
        auto encoder_3d_1 = this->encoder_3d_1->forward(feature);
        auto decoder_1    = this->decoder_1   ->forward(encoder_2d_1, encoder_3d_1);
        return feature.slice(1, LFR_VIDEO_QUEUE_SIZE - 1, LFR_VIDEO_QUEUE_SIZE).squeeze(1).add(decoder_1);
    }

};

TORCH_MODULE(WudaoziModule);

/**
 * 吴道子模型（视频生成）
 */
class WudaoziModel : public lifuren::Model<torch::optim::Adam, lifuren::image::WudaoziModule, lifuren::dataset::SeqDatasetLoader> {

public:
    WudaoziModel(lifuren::config::ModelParams params = {}) : Model(params) {
    }
    ~WudaoziModel() {

    }
    
public:
    void defineDataset() override {
        if(lifuren::file::exists(this->params.train_path)) {
            this->trainDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.train_path);
        }
        if(lifuren::file::exists(this->params.val_path)) {
            this->valDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.val_path);
        }
        if(lifuren::file::exists(this->params.test_path)) {
            this->testDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.test_path);
        }
    }
    void defineOptimizer() override {
        torch::optim::AdamOptions optims;
        optims.lr(this->params.lr);
        optims.eps(0.0001);
        this->optimizer = std::make_unique<torch::optim::Adam>(this->model->parameters(), optims);
    }
    torch::Tensor loss(torch::Tensor& label, torch::Tensor& pred) override {
        // L1Loss
        // MSELoss
        // HuberLoss
        // SmoothL1Loss
        // CrossEntropyLoss
        // return torch::smooth_l1_loss(pred, label);
        return torch::sum((pred - label).abs(), { 1, 2, 3 }, true).mean();
        // return torch::sum((pred - label).pow(2), { 1, 2, 3 }, true).mean();
    }

};

} // END OF lifuren::image

#endif // END OF LFR_HEADER_CORE_IMAGE_MODEL_HPP
