#include "lifuren/Wudaozi.hpp"

#include <cmath>
#include <random>

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "ATen/autocast_mode.h"

#include "lifuren/File.hpp"
#include "lifuren/Layer.hpp"
#include "lifuren/Trainer.hpp"

namespace lifuren {

    
/**
 * 动作模型
 * 
 * 根据图片生成动作向量
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
        int n_block = 2, int n_groups = 32, int attn_resolution = 32, const std::vector<int>& scales = { 1, 2, 2, 4, 4 }) {
        this->head = this->register_module("head", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, embedding_channels, { 3, 3 }).padding(1)));
        int min_img_size = std::min(img_height, img_width);
        torch::OrderedDict<std::string, std::shared_ptr<Module>> encoder_blocks;
        std::vector<std::tuple<int, int>> encoder_channels;
        int cur_c = embedding_channels;
        auto skip_pooling = 0;
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

/**
 * 吴道子模型（视频生成）
 * 
 * step_embed + unet = 图片生成
 * time_embed + pose = 动作向量
 * time_embed + pose + pose_embed + vnet = 下一帧变化
 * time_embed + pose + pose_embed + vnet + step_embed + unet = 下一帧图片
 */
class WudaoziImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams params;
    lifuren::nn::TimeEmbedding time_embed{ nullptr }; // 姿势时间嵌入
    lifuren::nn::StepEmbedding step_embed{ nullptr }; // 步数嵌入
    lifuren::nn::PoseEmbedding pose_embed{ nullptr }; // 姿势嵌入
    Pose pose { nullptr }; // 姿势模型
    UNet unet { nullptr }; // 图片生成模型
    UNet vnet { nullptr }; // 视频预测模型

    torch::Tensor alpha;
    torch::Tensor beta;
    torch::Tensor sigma;
    torch::Tensor bar_alpha;
    torch::Tensor bar_beta;

    torch::Tensor bar_alpha_;
    torch::Tensor bar_alpha_pre_;
    torch::Tensor bar_beta_;
    torch::Tensor bar_beta_pre_;
    torch::Tensor alpha_;
    torch::Tensor sigma_;
    torch::Tensor epsilon_;

    int stride = 4;
    float eta = 1.0;
    int T = 1000;
    
public:
    WudaoziImpl(lifuren::config::ModelParams params = {}) : params(params) {
        int model_channels = 8; // 嵌入输入维度
        int embedding_channels = 64; // 嵌入输出维度
        this->pose = this->register_module("pose", Pose(3, model_channels, embedding_channels));
        this->unet = this->register_module("unet", UNet(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 3, embedding_channels));
        this->vnet = this->register_module("vnet", UNet(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 3, embedding_channels));
        this->step_embed = this->register_module("step_embed", lifuren::nn::TimeEmbedding(1000, model_channels, embedding_channels));
        this->pose_embed = this->register_module("pose_embed", lifuren::nn::PoseEmbedding(      4 * 8, embedding_channels));;
        this->time_embed = this->register_module("time_embed", lifuren::nn::TimeEmbedding(LFR_VIDEO_FRAME_MAX, model_channels, embedding_channels));

        alpha = register_buffer("alpha", torch::sqrt(1.0 - 0.02 * torch::arange(1, T + 1) / (double) T));
        beta = register_buffer("beta", torch::sqrt(1.0 - torch::pow(alpha, 2)));
        bar_alpha = register_buffer("bar_alpha", torch::cumprod(alpha, 0));
        bar_beta = register_buffer("bar_beta", torch::sqrt(1.0 - bar_alpha.pow(2)));
        sigma = register_buffer("sigma", beta.clone());

        bar_alpha_ = bar_alpha.index({torch::indexing::Slice({torch::indexing::None, torch::indexing::None, stride})});
        bar_alpha_pre_ = torch::pad(bar_alpha_.index({torch::indexing::Slice(torch::indexing::None, -1)}), {1, 0}, "constant", 1);
        bar_beta_ = torch::sqrt(1.0 - torch::pow(bar_alpha_, 2));
        bar_beta_pre_ = torch::sqrt(1.0 - torch::pow(bar_alpha_pre_, 2));
        alpha_ = bar_alpha_ / bar_alpha_pre_;
        sigma_ = bar_beta_pre_ / bar_beta_ * torch::sqrt(1.0 - torch::pow(alpha_, 2)) * eta;
        epsilon_ = bar_beta_ - alpha_ * torch::sqrt(torch::pow(bar_beta_pre_, 2) - torch::pow(sigma_, 2));
    
        register_buffer("bar_alpha_", bar_alpha_);
        register_buffer("bar_alpha_pre_", bar_alpha_pre_);
        register_buffer("bar_beta_", bar_beta_);
        register_buffer("bar_beta_pre_", bar_beta_pre_);
        register_buffer("alpha_", alpha_);
        register_buffer("sigma_", sigma_);
        register_buffer("epsilon_", epsilon_);
    }
    ~WudaoziImpl() {
        this->unregister_module("unet");
    }

public:
    torch::Tensor forward_pose(torch::Tensor image, torch::Tensor t) {
        t = this->time_embed->forward(t);
        return this->pose->forward(image, t);
    }
    torch::Tensor forward_vnet(torch::Tensor image, torch::Tensor pose) {
        pose = this->pose_embed->forward(pose);
        return this->vnet->forward(image, pose);
    }
    torch::Tensor forward_unet(torch::Tensor feature, torch::Tensor t) {
        t = this->step_embed->forward(t);
        return this->unet->forward(feature, t);
    }
    /**
     * 对图片添加指定噪声
     */
    torch::Tensor mask_noise(const torch::Tensor& batch_images, const torch::Tensor& batch_noise, const torch::Tensor& batch_steps) {
        auto batch_bar_alpha = (bar_alpha).index({batch_steps}).reshape({-1, 1, 1, 1});
        auto batch_bar_beta = (bar_beta).index({batch_steps}).reshape({-1, 1, 1, 1});
        auto batch_noise_images = batch_images * batch_bar_alpha + batch_noise * batch_bar_beta;
        return batch_noise_images;
    }
    /**
     * 对图片添加随机噪声
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> make_noise(const torch::Tensor& image) {
        const auto& batch_images = image;
        auto batch_size = batch_images.size(0);
        std::vector<int> steps(T);
        std::iota(steps.begin(), steps.end(), 0);
        std::shuffle(steps.begin(), steps.end(), std::mt19937(std::random_device()()));
        steps.resize(batch_size);
        auto batch_steps = torch::tensor(steps, torch::TensorOptions().device(lifuren::get_device()).dtype(torch::kLong));
        auto batch_noise = torch::randn_like(batch_images);
        auto batch_noise_images = this->mask_noise(batch_images, batch_noise, batch_steps);
        return std::make_tuple(batch_noise_images, batch_steps, batch_noise);
    }
    inline torch::Tensor pred_image(torch::Tensor z, int t0) {
        auto T_ = bar_alpha_.size(0);
        for (int i = t0; i < T_; i++) {
            auto t = T_ - i - 1;
            auto bt = torch::tensor({t * stride}, torch::TensorOptions().device(lifuren::get_device())).repeat(z.size(0));
            z = z - epsilon_.index({t}) * this->forward_unet(z, bt);
            z = z / alpha_.index({t});
            z = z + torch::randn_like(z) * sigma_.index({t});
        }
        auto x_samples = torch::clip(z, -1, 1); // (n * n, 3, h, w)
        return x_samples;
    }
    /**
     * 对随机噪点图片降噪
     */
    torch::Tensor pred_image(int img_height, int img_width, int n, int t0) {
        torch::NoGradGuard no_grad_guard;
        torch::Tensor z = torch::randn({n, 3, img_height, img_width}, torch::TensorOptions().device(lifuren::get_device())).repeat({ 2, 1, 1, 1 });
        return this->pred_image(z, t0);
    }
    /**
     * 对指定噪点图片降噪
     */
    torch::Tensor pred_image(int img_height, int img_width, int t0, torch::Tensor feature) {
        torch::NoGradGuard no_grad_guard;
        auto batch_noise = torch::randn_like(feature);
        auto batch_steps = torch::tensor({ t0 }, torch::TensorOptions().device(lifuren::get_device()).dtype(torch::kLong));
        auto z = this->mask_noise(feature, batch_noise, batch_steps);
        return this->pred_image(z, t0);
    }
    /**
     * 对指定噪点图片降噪
     */
    torch::Tensor pred_image(int img_height, int img_width, int t, int t0, torch::Tensor feature) {
        torch::NoGradGuard no_grad_guard;
        auto batch_noise = torch::randn_like(feature);
        auto batch_times = torch::tensor({ t  }, torch::TensorOptions().device(lifuren::get_device()).dtype(torch::kLong));
        auto batch_steps = torch::tensor({ t0 }, torch::TensorOptions().device(lifuren::get_device()).dtype(torch::kLong));
        auto z = this->mask_noise(feature, batch_noise, batch_steps);
        auto pose = this->forward_pose(z, batch_times);
        pose = this->forward_vnet(z, pose);
        z = z + pose;
        return this->pred_image(z, t0);
    }
};

TORCH_MODULE(Wudaozi);

/**
 * 吴道子模型训练器（视频生成）
 */
class WudaoziTrainer : public lifuren::Trainer<torch::optim::AdamW, lifuren::Wudaozi, lifuren::dataset::RndDatasetLoader> {

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
        torch::optim::AdamWOptions options;
        options.lr(this->params.lr);
        options.weight_decay(0.0001);
        this->optimizer = std::make_unique<torch::optim::AdamW>(this->model->parameters(), options);
    }
    void val(const size_t epoch) override {
        if(epoch % this->params.check_epoch == 1) {
            torch::NoGradGuard no_grad_guard;
            auto result = this->model->pred_image(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 4, 0);
            cv::Mat image(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 4, CV_8UC3);
            lifuren::dataset::image::tensor_to_mat(image, result.to(torch::kFloat32).to(torch::kCPU));
            auto path = lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "pred", std::to_string(epoch) + ".jpg" }).string();
            cv::imwrite(path, image);
            SPDLOG_INFO("保存图片：{}", path);
        }
    }
    void loss(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override {
        auto device_feature = feature.to(lifuren::get_device());
        auto source_image = device_feature.slice(1, 0, 1).squeeze(1);
        auto target_image = device_feature.slice(1, 1, 2).squeeze(1);
        auto [noise_images, steps, noise] = this->model->make_noise(target_image);
        auto denoise = this->model->forward_unet(noise_images, steps);
        loss = torch::mse_loss(denoise, noise);
        
        // auto noise_loss = torch::mse_loss(denoise, noise);
        // auto pose = this->model->forward_pose(source_image, {}); // TODO: 步数
        // auto pose_loss = torch::mse_loss(pose, label);
        // auto next = this->model->forward_vnet(source_image, pose);
        // auto next_loss = torch::mse_loss(next, target_image);
        // loss = noise_loss + pose_loss + next_loss;

        // loss = torch::sum((denoise - noise).pow(2), {1, 2, 3}, true).mean();
    }
    torch::Tensor pred(torch::Tensor feature, torch::Tensor t) {
        torch::NoGradGuard no_grad_guard;
        return this->model->pred_image(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 100, feature);
    }

};

template<typename T>
class WudaoziClientImpl : public ClientImpl<lifuren::config::ModelParams, std::string, std::string, T> {

public:
    std::tuple<bool, std::string> pred(const std::string& input) override;
    std::tuple<bool, std::string> predImage(const std::string& input);
    std::tuple<bool, std::string> predVideo(const std::string& input);

};

}; // END OF lifuren

template<>
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::predImage(const std::string& input) {
    // TODO
    return {};
}

template<>
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::predVideo(const std::string& input) {
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
    auto tensor = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0).to(lifuren::get_device());
    for(int i = 0; i < LFR_VIDEO_FRAME_SIZE; ++i) {
        // LFR_VIDEO_FRAME_MAX 取余
        auto result = this->trainer->pred(tensor, torch::tensor({ i }).to(lifuren::get_device()));
        lifuren::dataset::image::tensor_to_mat(image, result.to(torch::kFloat32).to(torch::kCPU));
        writer.write(image);
    }
    writer.release();
    return { true, output };
}

template<>
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::pred(const std::string& input) {
    if(!this->trainer) {
        return { false, {} };
    }
    if(lifuren::file::is_file(input)) {
        return this->predVideo(input);
    } else if(lifuren::file::is_folder(input)) {
        return this->predImage(input);
    } else {
        return { false, {} };
    }
}

std::unique_ptr<lifuren::WudaoziClient> lifuren::get_wudaozi_client() {
    return std::make_unique<lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>>();
}
