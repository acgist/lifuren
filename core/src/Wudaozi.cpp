#include "lifuren/Wudaozi.hpp"

#include <cmath>
#include <random>

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Layer.hpp"
#include "lifuren/Trainer.hpp"

namespace lifuren {

/**
 * 训练模式
 */
enum class TrainType {

POSE,
UNET,
VNET,
ALL

};

namespace nn {

/**
 * 姿势矩阵模型
 * 
 * 姿势矩阵生成方式：
 * 1. 通过已有视频（视频姿势风格迁移）
 * 2. 通过图片生成
 * 3. 通过音频生成
 */
class PoseImpl : public torch::nn::Module {

private:
    lifuren::nn::ResidualBlock  res_1 { nullptr };
    lifuren::nn::ResidualBlock  res_2 { nullptr };
    lifuren::nn::Downsample     down_1{ nullptr };
    lifuren::nn::Downsample     down_2{ nullptr };
    lifuren::nn::AttentionBlock attn  { nullptr };
    torch::nn::Sequential       pose  { nullptr };

public:
    PoseImpl(int channels, int res_embedding_dims, int num_groups = 8, int attn_embedding_dims = LFR_VIDEO_POSE_WIDTH * LFR_VIDEO_POSE_HEIGHT) {
        SPDLOG_INFO("pos channels = {} res_embedding_dims = {} num_groups = {} attn_embedding_dims = {}", channels, res_embedding_dims, num_groups, attn_embedding_dims);
        this->res_1  = this->register_module("res_1",  lifuren::nn::ResidualBlock(channels,  8, res_embedding_dims, num_groups));
        this->res_2  = this->register_module("res_2",  lifuren::nn::ResidualBlock(8,        16, res_embedding_dims, num_groups));
        this->down_1 = this->register_module("down_1", lifuren::nn::Downsample( 8, num_groups, 4));
        this->down_2 = this->register_module("down_2", lifuren::nn::Downsample(16, num_groups, 8));
        this->attn   = this->register_module("attn",   lifuren::nn::AttentionBlock(16, 8, attn_embedding_dims, num_groups));
        this->pose   = this->register_module("pose",   torch::nn::Sequential(
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, 16)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 1, 3).padding(1).bias(false))
        ));
    }
    ~PoseImpl() {
        this->unregister_module("res_1");
        this->unregister_module("res_2");
        this->unregister_module("down_1");
        this->unregister_module("down_2");
        this->unregister_module("attn");
        this->unregister_module("pose");
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

} // END OF lifuren::nn
    
/**
 * 吴道子模型（视频生成）
 * 
 * 上一帧图片 + 加噪 = 上一帧噪声图片 + 姿势矩阵 = 下一帧噪声图片 -> 降噪 = 下一帧图片
 * 
 * step_embed + unet = 图片生成
 * time_embed + pose = 姿势矩阵
 * time_embed + pose + pose_embed + vnet = 下一帧变化
 * time_embed + pose + pose_embed + vnet + step_embed + unet = 下一帧图片
 */
class WudaoziImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams params;
    lifuren::nn::TimeEmbedding time_embed{ nullptr }; // 时间嵌入
    lifuren::nn::StepEmbedding step_embed{ nullptr }; // 步数嵌入
    lifuren::nn::PoseEmbedding pose_embed{ nullptr }; // 姿势嵌入
    lifuren::nn::Pose pose { nullptr }; // 姿势模型
    lifuren::nn::UNet unet { nullptr }; // 图片生成模型
    lifuren::nn::UNet vnet { nullptr }; // 视频预测模型

    torch::Tensor alpha;
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
        alpha = register_buffer("alpha", torch::sqrt(1.0 - 0.02 * torch::arange(1, T + 1) / (double) T));
        bar_alpha = register_buffer("bar_alpha", torch::cumprod(alpha, 0));
        bar_beta = register_buffer("bar_beta", torch::sqrt(1.0 - bar_alpha.pow(2)));

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

        int model_channels = 8; // 嵌入输入维度
        int embedding_dims = 64; // 嵌入输出维度
        this->pose = this->register_module("pose", lifuren::nn::Pose(3, embedding_dims, 8));
        this->unet = this->register_module("unet", lifuren::nn::UNet(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, 3, embedding_dims));
        this->vnet = this->register_module("vnet", lifuren::nn::UNet(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, 3, embedding_dims));
        this->step_embed = this->register_module("step_embed", lifuren::nn::StepEmbedding(T, model_channels, embedding_dims));
        this->pose_embed = this->register_module("pose_embed", lifuren::nn::PoseEmbedding(      4 * 8, embedding_dims));;
        this->time_embed = this->register_module("time_embed", lifuren::nn::TimeEmbedding(LFR_VIDEO_FRAME_MAX, model_channels, embedding_dims));
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
        pose = this->pose_embed->forward(pose.flatten(1, 2));
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
    torch::Tensor pred_image(int img_height, int img_width, int t0, int n) {
        torch::NoGradGuard no_grad_guard;
        torch::Tensor z = torch::randn({n, 3, img_height, img_width}, torch::TensorOptions().device(lifuren::get_device()));
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
    torch::Tensor pred_image(int img_height, int img_width, int t0, int t, torch::Tensor feature) {
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

private:
    int count = 0;
    double pose_loss = 0.0;
    double unet_loss = 0.0;
    double vnet_loss = 0.0;
    TrainType train_type = TrainType::ALL;

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
        if(this->count != 0) {
            SPDLOG_INFO(
                "姿势损失：{:.6f}，视频损失：{:.6f}，图片损失：{:.6f}。",
                this->pose_loss / this->count,
                this->vnet_loss / this->count,
                this->unet_loss / this->count
            );
            this->count = 0;
            this->pose_loss = 0.0;
            this->vnet_loss = 0.0;
            this->unet_loss = 0.0;
        }
        if(epoch % this->params.check_epoch == 1) {
            torch::NoGradGuard no_grad_guard;
            if(this->train_type == TrainType::UNET || this->train_type == TrainType::ALL) {
                auto result = this->model->pred_image(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 0, 2 * 4);
                cv::Mat image(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 4, CV_8UC3);
                lifuren::dataset::image::tensor_to_mat(image, result.to(torch::kFloat32).to(torch::kCPU));
                auto path = lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "pred", std::to_string(epoch) + ".jpg" }).string();
                cv::imwrite(path, image);
                SPDLOG_INFO("保存图片：{}", path);
            }
        }
    }
    void loss(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override {
        torch::Tensor time, pose, source_image, target_image, target_noise, step, noise, source_noise;
        {
            torch::NoGradGuard no_grad_guard;
            auto device = lifuren::get_device();
            time = label.slice(1, 0, 1).squeeze(1).select(1, 0).select(1, 0).to(torch::kLong).to(device);
            pose = label.slice(1, 1, 2).squeeze(1).to(device);
            source_image = feature.slice(1, 0, 1).squeeze(1).to(device);
            target_image = feature.slice(1, 1, 2).squeeze(1).to(device);
            std::tie(target_noise, step, noise) = this->model->make_noise(target_image);
            source_noise = this->model->mask_noise(source_image, noise, step);
        }
        // loss = torch::sum((denoise - noise).pow(2), {1, 2, 3}, true).mean();
        torch::Tensor noise_loss, pose_loss, next_loss;
        if(train_type == TrainType::UNET || train_type == TrainType::ALL) {
            auto denoise = this->model->forward_unet(target_noise, step);
            noise_loss = torch::mse_loss(denoise, noise);
            this->unet_loss += noise_loss.template item<float>();
        }
        if(train_type == TrainType::POSE || train_type == TrainType::ALL) {
            auto pred_pose = this->model->forward_pose(source_noise, time);
            pose_loss = torch::mse_loss(pred_pose, pose.unsqueeze(1));
            this->pose_loss += pose_loss.template item<float>();
        }
        if(train_type == TrainType::VNET || train_type == TrainType::ALL) {
            auto next = this->model->forward_vnet(source_noise, pose);
            next_loss = torch::mse_loss(next, target_noise);
            this->vnet_loss += next_loss.template item<float>();
        }
        if(train_type == TrainType::UNET) {
            loss = noise_loss;
        } else if(train_type == TrainType::POSE) {
            loss = pose_loss;
        } else if(train_type == TrainType::VNET) {
            loss = next_loss;
        } else {
            loss = noise_loss + pose_loss + next_loss;
        }
        ++this->count;
    }
    torch::Tensor pred(int n) {
        torch::NoGradGuard no_grad_guard;
        return this->model->pred_image(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 0, n);
    }
    torch::Tensor pred(torch::Tensor feature, int t) {
        torch::NoGradGuard no_grad_guard;
        // return this->model->pred_image(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 100, feature);
        return this->model->pred_image(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 100, t, feature);
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
    cv::Mat image(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, CV_8UC3);
    auto result = this->trainer->pred(1);
    lifuren::dataset::image::tensor_to_mat(image, result.to(torch::kFloat32).to(torch::kCPU));
    auto path = lifuren::file::join({ input, std::to_string(lifuren::config::uuid()) + ".jpg" }).string();
    cv::imwrite(path, image);
    return { true, path };
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
    // writer.write(image);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0).to(lifuren::get_device());
    for(int i = 0; i < LFR_VIDEO_FRAME_SIZE; ++i) {
        auto result = this->trainer->pred(tensor, i % LFR_VIDEO_FRAME_MAX);
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
