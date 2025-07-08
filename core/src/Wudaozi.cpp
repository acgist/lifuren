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
    VNET,
    INET,
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
 * pose_time_embedding + pose = 姿势生成
 * vnet_pose_embedding + vnet = 视频生成
 * inet_step_embedding + inet = 图片生成
 */
class WudaoziImpl : public torch::nn::Module {

private:
    int   T      = 1000; // DDPM步数
    int   stride = 4;    // DDIM步幅
    float eta    = 1.0;  // DDIM随机

    lifuren::config::ModelParams params;
    
    lifuren::nn::Pose pose{ nullptr }; // 姿势模型
    lifuren::nn::UNet vnet{ nullptr }; // 视频模型
    lifuren::nn::UNet inet{ nullptr }; // 图片模型
    lifuren::nn::TimeEmbedding pose_time_embedding{ nullptr }; // 姿势嵌入
    lifuren::nn::PoseEmbedding vnet_pose_embedding{ nullptr }; // 视频嵌入
    lifuren::nn::StepEmbedding inet_step_embedding{ nullptr }; // 图片嵌入

    torch::Tensor alpha;
    torch::Tensor bar_alpha;
    torch::Tensor bar_alpha_;
    torch::Tensor bar_alpha_pre_;
    torch::Tensor bar_beta;
    torch::Tensor bar_beta_;
    torch::Tensor bar_beta_pre_;
    torch::Tensor alpha_;
    torch::Tensor sigma_;
    torch::Tensor epsilon_;

public:
    WudaoziImpl(lifuren::config::ModelParams params = {}) : params(params) {
        this->alpha          = this->register_buffer("alpha", torch::sqrt(1.0 - 0.02 * torch::arange(1, this->T + 1) / (double) this->T));
        this->bar_alpha      = this->register_buffer("bar_alpha",      torch::cumprod(this->alpha, 0));
        this->bar_alpha_     = this->register_buffer("bar_alpha_",     this->bar_alpha.index({ torch::indexing::Slice({ torch::indexing::None, torch::indexing::None, this->stride }) }));
        this->bar_alpha_pre_ = this->register_buffer("bar_alpha_pre_", torch::pad(this->bar_alpha_.index({ torch::indexing::Slice(torch::indexing::None, -1) }), { 1, 0 }, "constant", 1));
        this->bar_beta       = this->register_buffer("bar_beta",      torch::sqrt(1.0 - torch::pow(this->bar_alpha,      2)));
        this->bar_beta_      = this->register_buffer("bar_beta_",     torch::sqrt(1.0 - torch::pow(this->bar_alpha_,     2)));
        this->bar_beta_pre_  = this->register_buffer("bar_beta_pre_", torch::sqrt(1.0 - torch::pow(this->bar_alpha_pre_, 2)));
        this->alpha_   = this->register_buffer("alpha_",   this->bar_alpha_ / this->bar_alpha_pre_);
        this->sigma_   = this->register_buffer("sigma_",   this->bar_beta_pre_ / this->bar_beta_ * torch::sqrt(1.0 - torch::pow(this->alpha_, 2)) * this->eta);
        this->epsilon_ = this->register_buffer("epsilon_", this->bar_beta_ - this->alpha_ * torch::sqrt(torch::pow(this->bar_beta_pre_, 2) - torch::pow(this->sigma_, 2)));
        int image_channels     =  3; // 图片输入维度
        int embedding_in_dims  =  8; // 嵌入输入维度
        int embedding_out_dims = 64; // 嵌入输出维度
        this->pose = this->register_module("pose", lifuren::nn::Pose(image_channels, embedding_out_dims));
        this->vnet = this->register_module("vnet", lifuren::nn::UNet(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, image_channels, embedding_out_dims));
        this->inet = this->register_module("inet", lifuren::nn::UNet(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, image_channels, embedding_out_dims));
        this->pose_time_embedding = this->register_module("pose_time_embedding", lifuren::nn::TimeEmbedding(LFR_VIDEO_FRAME_MAX, embedding_in_dims, embedding_out_dims));
        this->vnet_pose_embedding = this->register_module("vnet_pose_embedding", lifuren::nn::PoseEmbedding(LFR_VIDEO_POSE_WIDTH * LFR_VIDEO_POSE_HEIGHT, embedding_out_dims));
        this->inet_step_embedding = this->register_module("inet_step_embedding", lifuren::nn::StepEmbedding(this->T, embedding_in_dims, embedding_out_dims));
    }
    ~WudaoziImpl() {
        this->unregister_module("pose");
        this->unregister_module("vnet");
        this->unregister_module("inet");
        this->unregister_module("pose_time_embedding");
        this->unregister_module("vnet_pose_embedding");
        this->unregister_module("inet_step_embedding");
    }

public:
    torch::Tensor forward_pose(torch::Tensor image, torch::Tensor time) {
        return this->pose->forward(image, this->pose_time_embedding->forward(time));
    }
    torch::Tensor forward_vnet(torch::Tensor image, torch::Tensor pose) {
        return this->vnet->forward(image, this->vnet_pose_embedding->forward(pose.flatten(1, 2)));
    }
    torch::Tensor forward_inet(torch::Tensor feature, torch::Tensor step) {
        return this->inet->forward(feature, this->inet_step_embedding->forward(step));
    }
    torch::Tensor mask_noise(const torch::Tensor& batch_images, const torch::Tensor& batch_steps, const torch::Tensor& batch_noises) {
        auto batch_bar_alpha = this->bar_alpha.index({ batch_steps }).reshape({ -1, 1, 1, 1 });
        auto batch_bar_beta  = this->bar_beta .index({ batch_steps }).reshape({ -1, 1, 1, 1 });
        return batch_images * batch_bar_alpha + batch_noises * batch_bar_beta;
    }
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> make_noise(const torch::Tensor& batch_images) {
        std::vector<int> steps(this->T);
        std::iota(steps.begin(), steps.end(), 0);
        std::shuffle(steps.begin(), steps.end(), std::mt19937(std::random_device()()));
        steps.resize(batch_images.size(0));
        auto batch_steps  = torch::tensor(steps).to(lifuren::get_device()).to(torch::kLong);
        auto batch_noises = torch::randn_like(batch_images);
        auto batch_noise_images = this->mask_noise(batch_images, batch_steps, batch_noises);
        return std::make_tuple(batch_noise_images, batch_steps, batch_noises);
    }
    inline torch::Tensor denoise(torch::Tensor z, int t0) {
        auto T_ = this->bar_alpha_.size(0);
        auto device = lifuren::get_device();
        for (int i = t0; i < T_; ++i) {
            auto t = T_ - i - 1;
            auto x = torch::tensor({ t * this->stride }).to(device).repeat(z.size(0));
            z = z - this->epsilon_.index({ t }) * this->forward_inet(z, x);
            z = z / this->alpha_.index({ t });
            z = z + torch::randn_like(z) * this->sigma_.index({ t });
        }
        return torch::clip(z, -1, 1);
    }
    torch::Tensor pred_image(int n, int height, int width, int t0) {
        torch::NoGradGuard no_grad_guard;
        return this->denoise(torch::randn({ n, 3, height, width }).to(lifuren::get_device()), t0);
    }
    torch::Tensor pred_image(torch::Tensor images, int t0) {
        torch::NoGradGuard no_grad_guard;
        auto batch_steps  = torch::tensor({ t0 }).to(lifuren::get_device()).to(torch::kLong);
        auto batch_noises = torch::randn_like(images);
        return this->denoise(this->mask_noise(images, batch_steps, batch_noises), t0);
    }
    torch::Tensor pred_image(torch::Tensor images, int t, int t0) {
        torch::NoGradGuard no_grad_guard;
        auto batch_times  = torch::tensor({ t  }).to(lifuren::get_device()).to(torch::kLong);
        auto batch_steps  = torch::tensor({ t0 }).to(lifuren::get_device()).to(torch::kLong);
        auto batch_noises = torch::randn_like(images);
        auto noise_images = this->mask_noise(images, batch_steps, batch_noises);
        return this->denoise(this->forward_vnet(noise_images, this->forward_pose(noise_images, batch_times)), t0);
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
    double vnet_loss = 0.0;
    double inet_loss = 0.0;
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
                this->inet_loss / this->count
            );
            this->count = 0;
            this->pose_loss = 0.0;
            this->vnet_loss = 0.0;
            this->inet_loss = 0.0;
        }
        if(epoch % this->params.check_epoch == 1) {
            torch::NoGradGuard no_grad_guard;
            if(this->train_type == TrainType::INET || this->train_type == TrainType::ALL) {
                auto result = this->model->pred_image(2 * 4, LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 0);
                cv::Mat image(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 4, CV_8UC3);
                lifuren::dataset::image::tensor_to_mat(image, result.to(torch::kFloat32).to(torch::kCPU));
                auto path = lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "pred", std::to_string(epoch) + ".jpg" }).string();
                cv::imwrite(path, image);
                SPDLOG_INFO("保存图片：{}", path);
            }
        }
    }
    void loss(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override {
        torch::Tensor time, pose, batch_steps, batch_noises, batch_prev_images, batch_next_images, batch_next_noise_images, batch_prev_noise_images;
        {
            torch::NoGradGuard no_grad_guard;
            auto device = lifuren::get_device();
            time = label.slice(1, 0, 1).squeeze(1).select(1, 0).select(1, 0).to(torch::kLong).to(device);
            pose = label.slice(1, 1, 2).squeeze(1).to(device);
            batch_prev_images = feature.slice(1, 0, 1).squeeze(1).to(device);
            batch_next_images = feature.slice(1, 1, 2).squeeze(1).to(device);
            std::tie(batch_next_noise_images, batch_steps, batch_noises) = this->model->make_noise(batch_next_images);
            batch_prev_noise_images = this->model->mask_noise(batch_prev_images, batch_steps, batch_noises);
        }
        // loss = torch::sum((denoise - noise).pow(2), { 1, 2, 3 }, true).mean();
        torch::Tensor pose_loss, vnet_loss, inet_loss;
        if(train_type == TrainType::POSE || train_type == TrainType::ALL) {
            auto pred_pose = this->model->forward_pose(batch_prev_noise_images, time);
            pose_loss = torch::mse_loss(pred_pose, pose.unsqueeze(1));
            this->pose_loss += pose_loss.template item<float>();
        }
        if(train_type == TrainType::VNET || train_type == TrainType::ALL) {
            auto pred_vnet = this->model->forward_vnet(batch_prev_noise_images, pose);
            vnet_loss = torch::mse_loss(pred_vnet, batch_next_noise_images);
            this->vnet_loss += vnet_loss.template item<float>();
        }
        if(train_type == TrainType::INET || train_type == TrainType::ALL) {
            auto pred_inet = this->model->forward_inet(batch_next_noise_images, batch_steps);
            inet_loss = torch::mse_loss(pred_inet, batch_noises);
            this->inet_loss += inet_loss.template item<float>();
        }
        if(train_type == TrainType::POSE) {
            loss = pose_loss;
        } else if(train_type == TrainType::VNET) {
            loss = vnet_loss;
        } else if(train_type == TrainType::INET) {
            loss = inet_loss;
        } else {
            loss = pose_loss + vnet_loss + inet_loss;
        }
        ++this->count;
    }
    torch::Tensor pred(int n) {
        return this->model->pred_image(n, LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 0);
    }
    torch::Tensor pred(torch::Tensor feature, int t0) {
        return this->model->pred_image(feature, t0);
    }
    torch::Tensor pred(torch::Tensor feature, int t, int t0) {
        return this->model->pred_image(feature, t, t0);
    }

};

template<typename T>
class WudaoziClientImpl : public ClientImpl<lifuren::config::ModelParams, lifuren::WudaoziParams, std::string, T> {

public:
    std::tuple<bool, std::string> pred(const lifuren::WudaoziParams& input) override;
    std::tuple<bool, std::string> predImage(const std::string& file);
    std::tuple<bool, std::string> predImage(const std::string& path, int n);
    std::tuple<bool, std::string> predVideo(const std::string& file);

};

}; // END OF lifuren

template<>
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::predImage(const std::string& file) {
    auto image = cv::imread(file);
    if(image.empty()) {
        SPDLOG_INFO("打开文件失败：{}", file);
        return { false, {} };
    }
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0).to(lifuren::get_device());
    auto result = this->trainer->pred(tensor, 100);
    lifuren::dataset::image::tensor_to_mat(image, result.to(torch::kFloat32).to(torch::kCPU));
    const auto output = lifuren::file::modify_filename(file, ".jpg", "gen");
    cv::imwrite(output, image);
    return { true, output };
}

template<>
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::predImage(const std::string& path, int n) {
    cv::Mat image(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH * n, CV_8UC3);
    auto result = this->trainer->pred(n);
    lifuren::dataset::image::tensor_to_mat(image, result.to(torch::kFloat32).to(torch::kCPU));
    auto output = lifuren::file::join({ path, std::to_string(lifuren::config::uuid()) + ".jpg" }).string();
    cv::imwrite(output, image);
    return { true, output };
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
    writer.write(image);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0).to(lifuren::get_device());
    for(int i = 0; i < LFR_VIDEO_FRAME_SIZE; ++i) {
        auto result = this->trainer->pred(tensor, i % LFR_VIDEO_FRAME_MAX, 100);
        lifuren::dataset::image::tensor_to_mat(image, result.to(torch::kFloat32).to(torch::kCPU));
        writer.write(image);
    }
    writer.release();
    return { true, output };
}

template<>
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::pred(const lifuren::WudaoziParams& params) {
    if(!this->trainer) {
        return { false, {} };
    }
    if(params.type == WudaoziType::RESET) {
        return this->predImage(params.file);
    } else if(params.type == WudaoziType::IMAGE) {
        return this->predImage(params.path, params.size);
    } else if(params.type == WudaoziType::VIDEO) {
        return this->predVideo(params.file);
    } else {
        return { false, {} };
    }
}

std::unique_ptr<lifuren::WudaoziClient> lifuren::get_wudaozi_client() {
    return std::make_unique<lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>>();
}
