#include "lifuren/Test.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Audio.hpp"
#include "lifuren/AudioModel.hpp"

[[maybe_unused]] static void testTrain() {
    const std::string path = lifuren::config::CONFIG.tmp;
    lifuren::audio::ShikuangModel model({
        .lr         = 0.001F,
        .batch_size = 100,
        .epoch_size = 4,
        .model_name = "shikuang",
        .train_path = lifuren::file::join({path, "shikuang", lifuren::config::DATASET_TRAIN}).string(),
        .val_path   = lifuren::file::join({path, "shikuang", lifuren::config::DATASET_VAL  }).string(),
        .test_path  = lifuren::file::join({path, "shikuang", lifuren::config::DATASET_TEST }).string(),
    });
    model.define();
    model.trainValAndTest();
    model.save(lifuren::file::join({lifuren::config::CONFIG.tmp, "shikuang.pt"}).string());
}

[[maybe_unused]] static void testPred() {
    auto client = lifuren::audio::getAudioClient("shikuang");
    client->load(lifuren::file::join({lifuren::config::CONFIG.tmp, "shikuang.pt"}).string());
    auto [success, output] = client->pred(lifuren::file::join({lifuren::config::CONFIG.tmp, "shikuang.mp3"}).string());
    SPDLOG_INFO("输出结果：{} - {}", success, output);
}

LFR_TEST(
    testTrain();
    // testPred();
);
