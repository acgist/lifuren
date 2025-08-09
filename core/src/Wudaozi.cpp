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

    VNET,
    INET,
    ALL

};

/**
 * 吴道子模型（视频生成）
 * 
 * vnet_time_embedding + vnet = 视频生成
 * inet_step_embedding + inet = 图片生成
 */
class WudaoziImpl : public torch::nn::Module {

private:
    int   T      = 1000; // DDPM步数
    int   stride = 4;    // DDIM步幅
    float eta    = 1.0;  // DDIM随机
    
    lifuren::config::ModelParams params;

    torch::DeviceType device{ torch::DeviceType::CPU }; // 计算设备
    
    lifuren::nn::UNet vnet{ nullptr }; // 视频模型
    lifuren::nn::UNet inet{ nullptr }; // 图片模型
    lifuren::nn::TimeEmbedding vnet_time_embedding{ nullptr }; // 视频嵌入
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
    WudaoziImpl(lifuren::config::ModelParams params = {}) : params(params), device(lifuren::get_device()) {
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
        this->vnet = this->register_module("vnet", lifuren::nn::UNet(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, image_channels, embedding_out_dims));
        this->inet = this->register_module("inet", lifuren::nn::UNet(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, image_channels, embedding_out_dims));
        this->vnet_time_embedding = this->register_module("vnet_time_embedding", lifuren::nn::TimeEmbedding(LFR_VIDEO_FRAME_MAX, embedding_in_dims, embedding_out_dims));
        this->inet_step_embedding = this->register_module("inet_step_embedding", lifuren::nn::StepEmbedding(this->T,             embedding_in_dims, embedding_out_dims));
    }
    ~WudaoziImpl() {
        this->unregister_module("vnet");
        this->unregister_module("inet");
        this->unregister_module("vnet_time_embedding");
        this->unregister_module("inet_step_embedding");
    }

public:
    torch::Tensor forward_vnet(torch::Tensor image, torch::Tensor time) {
        return this->vnet->forward(image, this->vnet_time_embedding->forward(time));
    }
    torch::Tensor forward_inet(torch::Tensor image, torch::Tensor step) {
        return this->inet->forward(image, this->inet_step_embedding->forward(step));
    }
    torch::Tensor mark_noise(const torch::Tensor& batch_images, const torch::Tensor& batch_steps, const torch::Tensor& batch_noises) {
        auto batch_bar_alpha = this->bar_alpha.index({ batch_steps }).reshape({ -1, 1, 1, 1 });
        auto batch_bar_beta  = this->bar_beta .index({ batch_steps }).reshape({ -1, 1, 1, 1 });
        return batch_images * batch_bar_alpha + batch_noises * batch_bar_beta;
    }
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> make_noise(const torch::Tensor& batch_images) {
        std::vector<int> steps(this->T);
        std::iota(steps.begin(), steps.end(), 0);
        std::shuffle(steps.begin(), steps.end(), std::mt19937(std::random_device()()));
        steps.resize(batch_images.size(0));
        auto batch_steps  = torch::tensor(steps).to(this->device).to(torch::kLong);
        auto batch_noises = torch::randn_like(batch_images);
        auto batch_noise_images = this->mark_noise(batch_images, batch_steps, batch_noises);
        return std::make_tuple(batch_noise_images, batch_steps, batch_noises);
    }
    inline torch::Tensor denoise(torch::Tensor z, int t0) {
        auto T_ = this->bar_alpha_.size(0);
        for (int i = t0; i < T_; ++i) {
            auto t = T_ - i - 1;
            auto x = torch::tensor({ t * this->stride }).to(this->device).repeat(z.size(0));
            z = z - this->epsilon_.index({ t }) * this->forward_inet(z, x);
            z = z / this->alpha_.index({ t });
            z = z + torch::randn_like(z) * this->sigma_.index({ t });
        }
        return torch::clip(z, -1, 1);
    }
    torch::Tensor pred_image(int n, int height, int width, int t0) {
        torch::NoGradGuard no_grad_guard;
        return this->denoise(torch::randn({ n, 3, height, width }).to(this->device), t0);
    }
    torch::Tensor pred_image(torch::Tensor images, int t0) {
        torch::NoGradGuard no_grad_guard;
        auto batch_steps  = torch::tensor({ t0 }).to(this->device).to(torch::kLong);
        auto batch_noises = torch::randn_like(images);
        auto batch_noise_images = this->mark_noise(images, batch_steps, batch_noises);
        return this->denoise(batch_noise_images, t0);
    }
    torch::Tensor pred_image(torch::Tensor images, int t, int t0) {
        torch::NoGradGuard no_grad_guard;
        auto batch_times  = torch::tensor({ t  }).to(this->device).to(torch::kLong);
        auto batch_steps  = torch::tensor({ t0 }).to(this->device).to(torch::kLong);
        auto batch_noises = torch::randn_like(images);
        auto batch_noise_images = this->mark_noise(images, batch_steps, batch_noises);
        return this->denoise(this->forward_vnet(batch_noise_images, batch_times), t0);
    }
};

TORCH_MODULE(Wudaozi);

/**
 * 吴道子模型训练器（视频生成）
 */
class WudaoziTrainer : public lifuren::Trainer<torch::optim::AdamW, lifuren::Wudaozi, lifuren::dataset::RndDatasetLoader> {

private:
    int count = 0;
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
                "视频损失：{:.6f}，图片损失：{:.6f}。",
                this->vnet_loss / this->count,
                this->inet_loss / this->count
            );
            this->count = 0;
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
    void loss(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& /*pred*/, torch::Tensor& loss) override {
        torch::Tensor batch_times, batch_steps, batch_noises,
            batch_prev_images, batch_prev_noise_images,
            batch_next_images, batch_next_noise_images;
        {
            torch::NoGradGuard no_grad_guard;
            batch_times = label.to(torch::kLong).to(this->device);
            batch_prev_images = feature.slice(1, 0, 1).squeeze(1).to(this->device);
            batch_next_images = feature.slice(1, 1, 2).squeeze(1).to(this->device);
            std::tie(batch_prev_noise_images, batch_steps, batch_noises) = this->model->make_noise(batch_prev_images);
            batch_next_noise_images = this->model->mark_noise(batch_next_images, batch_steps, batch_noises);
        }
        torch::Tensor vnet_loss, inet_loss;
        if(train_type == TrainType::VNET || train_type == TrainType::ALL) {
            auto pred_vnet = this->model->forward_vnet(batch_prev_noise_images, batch_times);
            vnet_loss = torch::mse_loss(pred_vnet, batch_next_noise_images);
            this->vnet_loss += vnet_loss.template item<float>();
        }
        if(train_type == TrainType::INET || train_type == TrainType::ALL) {
            auto pred_inet = this->model->forward_inet(batch_prev_noise_images, batch_steps);
            inet_loss = torch::mse_loss(pred_inet, batch_noises);
            this->inet_loss += inet_loss.template item<float>();
        }
        if(train_type == TrainType::VNET) {
            loss = vnet_loss;
        } else if(train_type == TrainType::INET) {
            loss = inet_loss;
        } else {
            loss = vnet_loss + inet_loss;
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
    std::tuple<bool, std::string> predReset(const std::string& file, int t0 = 100);
    std::tuple<bool, std::string> predImage(const std::string& path, int n  = 1);
    std::tuple<bool, std::string> predVideo(const std::string& file, int t0 = 100, int frame = 120);

};

}; // END OF lifuren

template<>
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::predReset(const std::string& file, int t0) {
    auto image = cv::imread(file);
    if(image.empty()) {
        SPDLOG_INFO("打开文件失败：{}", file);
        return { false, {} };
    }
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0).to(this->trainer->device);
    auto result = this->trainer->pred(tensor, t0);
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
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::predVideo(const std::string& input, int t0, int frame) {
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
    auto tensor = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0).to(this->trainer->device);
    for(int i = 0; i < frame; ++i) {
        SPDLOG_DEBUG("当前帧数：{}", i);
        auto result = this->trainer->pred(tensor, i % LFR_VIDEO_FRAME_MAX, t0);
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
    if(params.n < 0 || params.n > 16 || params.t0 < 0 || params.t0 > 1000) {
        return { false, {} };
    }
    if(params.type == WudaoziType::RESET) {
        return this->predReset(params.file, params.t0);
    } else if(params.type == WudaoziType::IMAGE) {
        return this->predImage(params.path, params.n);
    } else if(params.type == WudaoziType::VIDEO) {
        return this->predVideo(params.file, params.t0, params.n);
    } else {
        return { false, {} };
    }
}

std::unique_ptr<lifuren::WudaoziClient> lifuren::get_wudaozi_client() {
    return std::make_unique<lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>>();
}
