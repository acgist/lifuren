#include "lifuren/Test.hpp"

#include "lifuren/audio/AudioModel.hpp"

[[maybe_unused]] static void testShikuang() {
    const std::string path = lifuren::config::CONFIG.tmp;
    lifuren::audio::ShikuangModel model({
        .lr         = 0.01F,
        .batch_size = 100,
        .epoch_count = 16,
        .model_name = "baicai",
        .train_path = lifuren::file::join({path, "baicai", lifuren::config::DATASET_TRAIN}).string(),
        .val_path   = lifuren::file::join({path, "baicai", lifuren::config::DATASET_VAL  }).string(),
        .test_path  = lifuren::file::join({path, "baicai", lifuren::config::DATASET_TEST }).string(),
    });
    model.define();
    model.trainValAndTest();
    model.save(lifuren::config::CONFIG.tmp, "shikuang.pt");
}

LFR_TEST(
    testShikuang();
);
