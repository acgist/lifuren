#include "lifuren/Test.hpp"

#include <random>

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/Wudaozi.hpp"

[[maybe_unused]] static void testTrain() {
    auto client = lifuren::get_wudaozi_client();
    const std::string path = lifuren::config::CONFIG.tmp;
    client->trainValAndTest({
        .lr          = 0.0002F, // 0.01F
        .grad_clip   = 10.0,
        .batch_size  = 20,
        .epoch_size  = 256,
        .check_point = true,
        .model_name  = "wudaozi",
        .model_path  = lifuren::file::join({path, "wudaozi", "checkpoint"                  }).string(),
        .train_path  = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_TRAIN}).string(),
        .val_path    = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_VAL  }).string(),
        .test_path   = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_TEST }).string(),
    });
    client->save(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.pt"}).string());
}

[[maybe_unused]] static void testLoad() {
    auto client = lifuren::get_wudaozi_client();
    const std::string path = lifuren::config::CONFIG.tmp;
    client->load(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "checkpoint", "wudaozi.checkpoint.31.ckpt"}).string(), {
        .lr          = 0.0001F, // 0.01F
        .grad_clip   = 10.0,
        .batch_size  = 20,
        .epoch_size  = 256,
        .check_point = true,
        .model_name  = "wudaozi",
        .model_path  = lifuren::file::join({path, "wudaozi", "checkpoint"                  }).string(),
        .train_path  = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_TRAIN}).string(),
        .val_path    = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_VAL  }).string(),
        .test_path   = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_TEST }).string(),
    }, true);
    client->trainValAndTest();
    client->save(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.pt"}).string());
}

[[maybe_unused]] static void testPred() {
    auto client = lifuren::get_wudaozi_client();
    client->load(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.pt"}).string());
    {
        auto [success, output] = client->pred({
            .t0   = 150,
            .file = lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "train.jpg"}).string(),
            .type = lifuren::WudaoziType::RESET
        });
        SPDLOG_INFO("输出结果：{} - {}", success, output);
    }
    {
        auto [success, output] = client->pred({
            .t0   = 150,
            .file = lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.jpg"}).string(),
            .type = lifuren::WudaoziType::RESET
        });
        SPDLOG_INFO("输出结果：{} - {}", success, output);
    }
    {
        auto [success, output] = client->pred({
            .n    = 4,
            .path = lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi"}).string(),
            .type = lifuren::WudaoziType::IMAGE
        });
        SPDLOG_INFO("输出结果：{} - {}", success, output);
    }
    {
        auto [success, output] = client->pred({
            .t0   = 150,
            .file = lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "train.jpg"}).string(),
            .type = lifuren::WudaoziType::VIDEO
        });
        SPDLOG_INFO("输出结果：{} - {}", success, output);
    }
    {
        auto [success, output] = client->pred({
            .t0   = 150,
            .file = lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.jpg"}).string(),
            .type = lifuren::WudaoziType::VIDEO
        });
        SPDLOG_INFO("输出结果：{} - {}", success, output);
    }
}

[[maybe_unused]] static void testPlay() {
    cv::VideoCapture video    (lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.mp4"    }).string());
    cv::VideoCapture video_gen(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi_gen.mp4"}).string());
    if(!video.isOpened() || !video_gen.isOpened()) {
        SPDLOG_WARN("打开视频失败");
        return;
    }
    cv::Mat frame;
    cv::Mat frame_gen;
    while(video.read(frame) && video_gen.read(frame_gen)) {
        lifuren::dataset::image::resize(frame,     LFR_IMAGE_WIDTH * 1, LFR_IMAGE_HEIGHT * 1);
        lifuren::dataset::image::resize(frame,     LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        lifuren::dataset::image::resize(frame_gen, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        cv::imshow("frame",     frame);
        cv::imshow("frame_gen", frame_gen);
        if(cv::waitKey(60'000) == 27) {
            break;
        }
    }
}

[[maybe_unused]] static void testNoise() {
    int   T      = 1000;
    int   stride = 4;
    float eta    = 1.0;
    auto alpha = torch::sqrt(1.0 - 0.02 * torch::arange(1, T + 1) / (double) T);
    auto bar_alpha      = torch::cumprod(alpha, 0);
    auto bar_alpha_     = bar_alpha.index({ torch::indexing::Slice({ torch::indexing::None, torch::indexing::None, stride }) });
    auto bar_alpha_pre_ = torch::pad(bar_alpha_.index({ torch::indexing::Slice(torch::indexing::None, -1) }), { 1, 0 }, "constant", 1);
    auto bar_beta      = torch::sqrt(1.0 - torch::pow(bar_alpha, 2));
    auto bar_beta_     = torch::sqrt(1.0 - torch::pow(bar_alpha_, 2));
    auto bar_beta_pre_ = torch::sqrt(1.0 - torch::pow(bar_alpha_pre_, 2));
    auto alpha_   = bar_alpha_ / bar_alpha_pre_;
    auto sigma_   = bar_beta_pre_ / bar_beta_ * torch::sqrt(1.0 - torch::pow(alpha_, 2)) * eta;
    auto epsilon_ = bar_beta_ - alpha_ * torch::sqrt(torch::pow(bar_beta_pre_, 2) - torch::pow(sigma_, 2));
    auto image { cv::imread(lifuren::file::join({ lifuren::config::CONFIG.tmp, "image.jpg" }).string()) };
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    cv::imshow("image", image);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0);
    const auto& batch_images = torch::concat({ tensor, tensor, tensor, tensor, tensor, tensor }, 0);
    std::vector<int> steps{ 0, 10, 50, 100, 200, 999 };
    steps.resize(batch_images.size(0));
    auto batch_steps = torch::tensor(steps).to(torch::kLong);
    auto batch_bar_alpha = bar_alpha.index({ batch_steps }).reshape({ -1, 1, 1, 1 });
    auto batch_bar_beta  = bar_beta .index({ batch_steps }).reshape({ -1, 1, 1, 1 });
    auto batch_noises    = torch::randn_like(batch_images);
    auto batch_noise_images = batch_images * batch_bar_alpha + batch_noises * batch_bar_beta;
    cv::Mat output_image(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 3, CV_8UC3);
    cv::Mat output_noise(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 3, CV_8UC3);
    lifuren::dataset::image::tensor_to_mat(output_image, batch_noise_images.to(torch::kFloat32).to(torch::kCPU));
    lifuren::dataset::image::tensor_to_mat(output_noise, batch_noises       .to(torch::kFloat32).to(torch::kCPU));
    cv::imshow("output_image", output_image);
    cv::imshow("output_noise", output_noise);
    cv::waitKey();
}

LFR_TEST(
    testTrain();
    testPred();
    // testPlay();
    // testNoise();
);
