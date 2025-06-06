#include "lifuren/Test.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/Image.hpp"
#include "lifuren/ImageModel.hpp"

[[maybe_unused]] static void testTrain() {
    const std::string path = lifuren::config::CONFIG.tmp;
    lifuren::image::WudaoziModel model({
        .lr         = 0.001F,
        .batch_size = 100,
        .epoch_size = 4096,
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
    // cv::VideoCapture video(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.mp4"}).string());
    cv::VideoCapture video(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi_gen.mp4"}).string());
    if(!video.isOpened()) {
        SPDLOG_WARN("打开视频失败");
        return;
    }
    cv::Mat frame;
    while(video.read(frame)) {
        lifuren::dataset::image::resize(frame, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
        cv::imshow("frame", frame);
        if(cv::waitKey(60'000) == 27) {
            break;
        }
    }
}

LFR_TEST(
    testTrain();
    testPred();
    testPlay();
);
