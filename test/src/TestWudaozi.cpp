#include "lifuren/Test.hpp"

#include <random>

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/Wudaozi.hpp"

[[maybe_unused]] static void testTrain() {
    auto client = lifuren::get_wudaozi_client();
    client->trainValAndTest({
        .lr          = 0.0002F, // 0.01F
        .grad_clip   = 10.0,
        .batch_size  = 10,
        .epoch_size  = 16,
        .check_point = true,
        .model_name  = "wudaozi",
        .model_path  = lifuren::file::join({tmp_directory, "wudaozi", "checkpoint"                  }).string(),
        .train_path  = lifuren::file::join({tmp_directory, "wudaozi", lifuren::config::DATASET_TRAIN}).string(),
        .val_path    = lifuren::file::join({tmp_directory, "wudaozi", lifuren::config::DATASET_VAL  }).string(),
        .test_path   = lifuren::file::join({tmp_directory, "wudaozi", lifuren::config::DATASET_TEST }).string(),
    });
    client->save(lifuren::file::join({tmp_directory, "wudaozi", "wudaozi.pt"}).string());
}

[[maybe_unused]] static void testTuning() {
    auto client = lifuren::get_wudaozi_client();
    client->load(lifuren::file::join({tmp_directory, "wudaozi", "checkpoint", "wudaozi.checkpoint.31.ckpt"}).string(), {
        .lr          = 0.0001F, // 0.01F
        .grad_clip   = 10.0,
        .batch_size  = 10,
        .epoch_size  = 128,
        .check_point = true,
        .model_name  = "wudaozi",
        .model_path  = lifuren::file::join({tmp_directory, "wudaozi", "checkpoint"                  }).string(),
        .train_path  = lifuren::file::join({tmp_directory, "wudaozi", lifuren::config::DATASET_TRAIN}).string(),
        .val_path    = lifuren::file::join({tmp_directory, "wudaozi", lifuren::config::DATASET_VAL  }).string(),
        .test_path   = lifuren::file::join({tmp_directory, "wudaozi", lifuren::config::DATASET_TEST }).string(),
    }, true);
    client->trainValAndTest();
    client->save(lifuren::file::join({tmp_directory, "wudaozi", "wudaozi.pt"}).string());
}

[[maybe_unused]] static void testPred() {
    auto client = lifuren::get_wudaozi_client();
    client->load(lifuren::file::join({tmp_directory, "wudaozi", "wudaozi.pt"}).string());
    {
        auto [success, output] = client->pred({
            .n    = 4,
            .path = lifuren::file::join({tmp_directory, "wudaozi"}).string(),
            .type = lifuren::WudaoziType::IMAGE
        });
        SPDLOG_INFO("输出结果：{} - {}", success, output);
    }
    {
        auto [success, output] = client->pred({
            .n    = 24,
            .t0   = 100,
            .file = lifuren::file::join({tmp_directory, "wudaozi", "train.jpg"}).string(),
            .type = lifuren::WudaoziType::VIDEO
        });
        SPDLOG_INFO("输出结果：{} - {}", success, output);
    }
    {
        auto [success, output] = client->pred({
            .n    = 24,
            .t0   = 100,
            .file = lifuren::file::join({tmp_directory, "wudaozi", "wudaozi.jpg"}).string(),
            .type = lifuren::WudaoziType::VIDEO
        });
        SPDLOG_INFO("输出结果：{} - {}", success, output);
    }
}

[[maybe_unused]] static void testPlay() {
    cv::VideoCapture video    (lifuren::file::join({tmp_directory, "wudaozi", "wudaozi.mp4"    }).string());
    cv::VideoCapture video_gen(lifuren::file::join({tmp_directory, "wudaozi", "wudaozi_gen.mp4"}).string());
    if(!video.isOpened() || !video_gen.isOpened()) {
        SPDLOG_WARN("打开视频失败");
        return;
    }
    cv::Mat frame_src;
    cv::Mat frame_gen;
    cv::Mat frame(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 4, CV_8UC3);
    while(video.read(frame_src) && video_gen.read(frame_gen)) {
        lifuren::dataset::image::resize(frame_src, LFR_IMAGE_WIDTH * 1, LFR_IMAGE_HEIGHT * 1);
        lifuren::dataset::image::resize(frame_src, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        lifuren::dataset::image::resize(frame_gen, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        frame_src.copyTo(frame(cv::Rect(LFR_IMAGE_WIDTH * 0, 0, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2)));
        frame_gen.copyTo(frame(cv::Rect(LFR_IMAGE_WIDTH * 2, 0, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2)));
        cv::imshow("frame", frame);
        if(cv::waitKey(60'000) == 27) {
            break;
        }
    }
}

[[maybe_unused]] static void testNoise() {
    // std::cout << lifuren::config::wudaozi::alpha << std::endl;
    // std::cout << lifuren::config::wudaozi::bar_alpha << std::endl;
    // std::cout << lifuren::config::wudaozi::bar_alpha_ << std::endl;
    std::cout << lifuren::config::wudaozi::bar_alpha_pre_ << std::endl;
    auto image { cv::imread(lifuren::file::join({ tmp_directory, "image.jpg" }).string()) };
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    cv::imshow("image", image);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0);
    const auto& batch_images = torch::concat({ tensor, tensor, tensor, tensor, tensor, tensor }, 0);
    std::vector<int> steps{ 0, 10, 50, 100, 200, 999 };
    steps.resize(batch_images.size(0));
    auto batch_steps        = torch::tensor(steps).to(torch::kLong);
    auto batch_bar_alpha    = lifuren::config::wudaozi::bar_alpha.index({ batch_steps }).reshape({ -1, 1, 1, 1 });
    auto batch_bar_beta     = lifuren::config::wudaozi::bar_beta .index({ batch_steps }).reshape({ -1, 1, 1, 1 });
    auto batch_noises       = torch::randn_like(batch_images);
    auto batch_noise_images = batch_images * batch_bar_alpha + batch_noises * batch_bar_beta;
    cv::Mat output_image(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 3, CV_8UC3);
    cv::Mat output_noise(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 3, CV_8UC3);
    lifuren::dataset::image::tensor_to_mat(output_noise, batch_noises      .to(torch::kFloat32).to(torch::kCPU));
    lifuren::dataset::image::tensor_to_mat(output_image, batch_noise_images.to(torch::kFloat32).to(torch::kCPU));
    cv::imshow("output_image", output_image);
    cv::imshow("output_noise", output_noise);
    cv::waitKey();
}

LFR_TEST(
    testTrain();
    // testTuning();
    testPred();
    // testPlay();
    // testNoise();
);
