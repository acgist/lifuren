#include "lifuren/Test.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/Image.hpp"
#include "lifuren/WudaoziModel.hpp"

[[maybe_unused]] static void testTrain() {
    const std::string path = lifuren::config::CONFIG.tmp;
    lifuren::image::WudaoziModel model({
        .lr         = 0.001F, // 0.01F
        .batch_size = 100,
        .epoch_size = 256,
        .model_name = "wudaozi",
        .train_path = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_TRAIN}).string(),
        .val_path   = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_VAL  }).string(),
        .test_path  = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_TEST }).string(),
    });
    model.define();
    model.trainValAndTest();
    model.save(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.pt"}).string());
}

[[maybe_unused]] static void testPred() {
    auto client = lifuren::image::getImageClient("wudaozi");
    client->load(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.pt"}).string(), {
        .batch_size = 1
    });
    auto [success, output] = client->pred(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.jpg"}).string());
    // auto [success, output] = client->pred(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.mp4"}).string());
    SPDLOG_INFO("输出结果：{} - {}", success, output);
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
        lifuren::dataset::image::resize(frame,     LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        lifuren::dataset::image::resize(frame_gen, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        cv::imshow("frame",     frame);
        cv::imshow("frame_gen", frame_gen);
        if(cv::waitKey(60'000) == 27) {
            break;
        }
    }
}

static void testLayer() {
    lifuren::image::AttentionBlock a(32, 1, 8);
    auto input = torch::randn({100, 32, 40, 20});
    auto output = a->forward(input);
    std::cout << input.sizes() << std::endl;
    std::cout << output.sizes() << std::endl;
    lifuren::image::Upsample upsample(32);
    output = upsample->forward(input);
    std::cout << output.sizes() << std::endl;
    lifuren::image::Downsample downsample(32);
    output = downsample->forward(input);
    std::cout << output.sizes() << std::endl;
    lifuren::image::ResidualBlock residual(32, 10);
    output = residual->forward(input, torch::randn({100, 8}));
    std::cout << output.sizes() << std::endl;
    system("pause");
}

LFR_TEST(
    // testTrain();
    // testPred();
    // testPlay();
    testLayer();
);
