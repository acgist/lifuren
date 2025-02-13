#include "lifuren/Test.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/video/Video.hpp"
#include "lifuren/video/VideoModel.hpp"

[[maybe_unused]] static void testTrain() {
    const std::string path = lifuren::config::CONFIG.tmp;
    lifuren::video::WudaoziModel model({
        .lr          = 0.001F,
        .batch_size  = 100,
        .epoch_count = 4,
        .model_name  = "video",
        .train_path  = lifuren::file::join({path, "video", lifuren::config::DATASET_TRAIN}).string(),
        .val_path    = lifuren::file::join({path, "video", lifuren::config::DATASET_VAL  }).string(),
        .test_path   = lifuren::file::join({path, "video", lifuren::config::DATASET_TEST }).string(),
    });
    model.define();
    model.trainValAndTest();
    model.save(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi.pt"}).string());
}

[[maybe_unused]] static void testPred() {
    auto client = lifuren::video::getVideoClient("video-wudaozi");
    client->load(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi.pt"}).string());
    lifuren::video::VideoParams params {
        .video = lifuren::file::join({lifuren::config::CONFIG.tmp, "video_short.mp4"}).string()
    };
    auto [success, output] = client->pred(params);
    SPDLOG_INFO("生成视频：{} - {}", success, output);
}

LFR_TEST(
    // testTrain();
    testPred();
);
