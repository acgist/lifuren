#include "lifuren/Wudaozi.hpp"

#include <cmath>

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Layer.hpp"
#include "lifuren/Trainer.hpp"

namespace lifuren {

/**
 * UNet
 */
class UNetImpl : public torch::nn::Module {

private:
    torch::nn::Conv2d head{ nullptr };
    lifuren::nn::TimeEmbedding time_embed{ nullptr };
    torch::nn::ModuleDict encoder_blocks{nullptr};
    torch::nn::ModuleDict decoder_blocks{nullptr};

public:
    UNetImpl(int img_height, int img_width, int channels, int model_channels, int embedding_channels,  int min_pixel = 4,
        int n_block = 2, int n_groups = 32, int attn_resolution = 32, const std::vector<int>& scales = { 2, 2, 2, 4, 4 }) {
        this->head = this->register_module("head", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, embedding_channels, { 3, 3 }).padding(1)));
        this->time_embed = this->register_module("time_embed", lifuren::nn::TimeEmbedding(LFR_VIDEO_FRAME_MAX, model_channels, embedding_channels));
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
                auto block = lifuren::nn::ResidualBlock(cur_c, scale * embedding_channels, embedding_channels, n_groups);
                cur_c = scale * embedding_channels;
                encoder_blocks.insert((std::stringstream() << "enc_block_" << i * n_block + j).str(), block.ptr());
            }
    
            if (min_img_size <= attn_resolution) {
                encoder_blocks.insert((std::stringstream() << "attn_enc_block_" << i * n_block).str(),
                lifuren::nn::AttentionBlock(cur_c, 8, img_height * img_width / std::pow(2, 2 * i), cur_c / 8).ptr());
            }
    
            // downsample block if not reach to `min_pixel`.
            if (min_img_size > min_pixel) {
                encoder_blocks.insert((std::stringstream() << "down_block_" << i).str(), lifuren::nn::Downsample(cur_c).ptr());
                min_img_size = min_img_size / 2;
            } else {
                skip_pooling += 1; // log how many times skip pooling.
            }
        }
        // mid
        encoder_blocks.insert((std::stringstream() << "enc_block_" << scales.size() * n_block).str(),
        lifuren::nn::ResidualBlock(cur_c, cur_c, embedding_channels, n_groups).ptr());
            // ?
            // lifuren::nn::ResidualBlock(ch, ch, time_embed_dim, dropout),
            // lifuren::nn::AttentionBlock(ch, num_heads=num_heads),
            // lifuren::nn::ResidualBlock(ch, ch, time_embed_dim, dropout)

        this->encoder_blocks = this->register_module("encoder_blocks", torch::nn::ModuleDict(encoder_blocks));

        std::reverse(encoder_channels.begin(), encoder_channels.end());

        torch::OrderedDict<std::string, std::shared_ptr<Module>> decoder_blocks;
        size_t m = 0;
        for (int i = scales.size() - 1; i > -1; i--) {
            auto rev_scale = scales[i];
            if (m >= skip_pooling) {
                decoder_blocks.insert((std::stringstream() << "up_block_" << m).str(), lifuren::nn::Upsample(cur_c).ptr());
                min_img_size *= 2;
            }

            for (size_t j = 0; j < n_block; j++) {
                int out_channels;
                int in_channels;
                std::tie(out_channels, in_channels) = encoder_channels[m * n_block + j];
                decoder_blocks.insert((std::stringstream() << "dec_block_" << m * n_block + j).str(),
                lifuren::nn::ResidualBlock(in_channels, out_channels, embedding_channels, n_groups).ptr());
                cur_c = out_channels;
            }

            if (min_img_size <= attn_resolution) {
                decoder_blocks.insert((std::stringstream() << "attn_dec_block_" << m * n_block).str(),
                lifuren::nn::AttentionBlock(cur_c, 8, img_height * img_width / std::pow(2, 2 * i), cur_c / 8).ptr());
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
        this->unregister_module("head");
        this->unregister_module("time_embed");
    }

public:
    torch::Tensor forward(torch::Tensor x, torch::Tensor t) {
        x = head(x);
        t = this->time_embed->forward(t);

        std::vector<torch::Tensor> inners;
    
        inners.push_back(x);
        for (const auto &item: encoder_blocks->items()) {
            auto name = item.first;
            auto module = item.second;
            // resudial block
            if (name.starts_with("enc")) {
                x = module->as<lifuren::nn::ResidualBlock>()->forward(x, t);
                inners.push_back(x);
            } else if (name.starts_with("attn")) {
                x = module->as<lifuren::nn::AttentionBlock>()->forward(x);
            }
                // downsample block
            else {
                x = module->as<lifuren::nn::Downsample>()->forward(x);
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
                x = module->as<lifuren::nn::Upsample>()->forward(x);
                torch::Tensor xi = inners_.back();
                inners_.pop_back(); // pop()
                x = x + xi;
            }
                // resudial block
            else if (name.starts_with("dec")) {
                torch::Tensor xi = inners_.back();
                inners_.pop_back(); // pop()
                x = module->as<lifuren::nn::ResidualBlock>()->forward(x, t);
                x = x + xi;
            } else if (name.starts_with("attn")) {
    
                x = module->as<lifuren::nn::AttentionBlock>()->forward(x);
            } else {
                x = module->as<torch::nn::Sequential>()->forward(x);
            }
        }
    
        return x;
    }

};

TORCH_MODULE(UNet);

/**
 * 吴道子模型（视频生成）
 */
class WudaoziImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams params;
    UNet unet { nullptr };
    
public:
    WudaoziImpl(lifuren::config::ModelParams params = {}) : params(params) {
        this->unet = this->register_module("unet", UNet(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 3, 8, 64));
    }
    ~WudaoziImpl() {
        this->unregister_module("unet");
    }

public:
    torch::Tensor forward(torch::Tensor feature, torch::Tensor t) {
        return this->unet->forward(feature, t);
    }

};

TORCH_MODULE(Wudaozi);

/**
 * 吴道子模型训练器（视频生成）
 */
class WudaoziTrainer : public lifuren::Trainer<torch::optim::Adam, lifuren::Wudaozi, lifuren::dataset::RndDatasetLoader> {

public:
    WudaoziTrainer(lifuren::config::ModelParams params = {}) : Trainer(params) {
    }
    ~WudaoziTrainer() {
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
        optims.eps(1e-5);
        this->optimizer = std::make_unique<torch::optim::Adam>(this->model->parameters(), optims);
    }
    void defineWeight() override {
        auto params = this->model->parameters();
        for(auto iter = params.begin(); iter != params.end(); ++iter) {
            if(iter->sizes().size() == 2) {
                torch::nn::init::xavier_uniform_(*iter);
            }
        }
    }
    void loss(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override {
        // L1Loss
        // MSELoss
        // HuberLoss
        // SmoothL1Loss
        // CrossEntropyLoss
        pred = this->model->forward(feature.slice(1, 0, 1).squeeze(1), label.squeeze(1));
        // loss = torch::mse_loss(pred, feature.slice(1, 1, 2).squeeze(1));
        loss = torch::smooth_l1_loss(pred, feature.slice(1, 1, 2).squeeze(1));
    }
    torch::Tensor pred(torch::Tensor feature, torch::Tensor t) {
        torch::NoGradGuard no_grad;
        return this->model->forward(feature, t);
    }

};

template<typename T>
class WudaoziClientImpl : public ClientImpl<lifuren::config::ModelParams, std::string, std::string, T> {

public:
    std::tuple<bool, std::string> pred(const std::string& input) override;

};

}; // END OF lifuren

template<>
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::pred(const std::string& input) {
    if(!this->trainer) {
        return { false, {} };
    }
    auto image = cv::imread(input);
    if(image.empty()) {
        SPDLOG_INFO("打开文件失败：{}", input);
        return { false, {} };
    }
    const auto output = lifuren::file::modify_filename(input, ".mp4", "gen");
    cv::VideoWriter writer(output, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), LFR_VIDEO_FPS, cv::Size(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT));
    if(!writer.isOpened()) {
        SPDLOG_WARN("视频文件打开失败：{}", output);
        return { false, output };
    }
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0).to(LFR_DTYPE).to(lifuren::get_device());
    for(int i = 0; i < LFR_VIDEO_FRAME_SIZE; ++i) {
        auto result = this->trainer->pred(tensor, torch::tensor({ i }).to(lifuren::get_device())).squeeze(0);
        lifuren::dataset::image::tensor_to_mat(image, result.to(torch::kFloat32).to(torch::kCPU));
        writer.write(image);
    }
    writer.release();
    return { true, output };
}

std::unique_ptr<lifuren::WudaoziClient> lifuren::get_wudaozi_client() {
    return std::make_unique<lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>>();
}
