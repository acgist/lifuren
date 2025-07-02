#include "lifuren/Test.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/Wudaozi.hpp"

[[maybe_unused]] static void testTrain() {
    auto client = lifuren::get_wudaozi_client();
    const std::string path = lifuren::config::CONFIG.tmp;
    client->trainValAndTest({
        .lr         = 0.0001F, // 0.01F
        .batch_size = 20,
        .epoch_size = 256,
        .model_name = "wudaozi",
        .train_path = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_TRAIN}).string(),
        .val_path   = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_VAL  }).string(),
        .test_path  = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_TEST }).string(),
    });
    client->save(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.pt"}).string());
}

[[maybe_unused]] static void testPred() {
    auto client = lifuren::get_wudaozi_client();
    client->load(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.pt"}).string(), {
        .batch_size = 1
    });
    {
        auto [success, output] = client->pred(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "lyf.jpg"}).string());
        SPDLOG_INFO("输出结果：{} - {}", success, output);
    }
    {
        auto [success, output] = client->pred(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi", "wudaozi.jpg"}).string());
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
        lifuren::dataset::image::resize(frame,     LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        lifuren::dataset::image::resize(frame_gen, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        cv::imshow("frame",     frame);
        cv::imshow("frame_gen", frame_gen);
        if(cv::waitKey(60'000) == 27) {
            break;
        }
    }
}

LFR_TEST(
    // testTrain();
    // testPred();
    testPlay();
);
