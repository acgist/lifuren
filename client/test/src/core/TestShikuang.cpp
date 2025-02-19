#include "lifuren/Test.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/audio/Audio.hpp"
#include "lifuren/audio/AudioModel.hpp"

[[maybe_unused]] static void testTrain() {
    const std::string path = lifuren::config::CONFIG.tmp;
    lifuren::audio::ShikuangModel model({
        .lr          = 0.001F,
        .batch_size  = LFR_DATASET_PCM_BATCH_SIZE,
        .epoch_count = 4,
        .model_name  = "baicai",
        .train_path  = lifuren::file::join({path, "baicai", lifuren::config::DATASET_TRAIN}).string(),
        .val_path    = lifuren::file::join({path, "baicai", lifuren::config::DATASET_VAL  }).string(),
        .test_path   = lifuren::file::join({path, "baicai", lifuren::config::DATASET_TEST }).string(),
    });
    model.define();
    model.trainValAndTest();
    model.save(lifuren::file::join({lifuren::config::CONFIG.tmp, "shikuang.pt"}).string());
}

[[maybe_unused]] static void testPred() {
    auto client = lifuren::audio::getAudioClient("audio-shikuang");
    client->load(lifuren::file::join({lifuren::config::CONFIG.tmp, "shikuang.pt"}).string());
    lifuren::audio::AudioParams params {
        // .audio = lifuren::file::join({lifuren::config::CONFIG.tmp, "tts.mp3"}).string()
        .audio = lifuren::file::join({lifuren::config::CONFIG.tmp, "baicai.mp3"}).string()
    };
    auto [success, output] = client->pred(params);
    SPDLOG_INFO("生成音频：{} - {}", success, output);
}

LFR_TEST(
    // testTrain();
    testPred();
);
