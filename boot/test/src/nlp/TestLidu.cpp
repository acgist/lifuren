#include "lifuren/Test.hpp"

#include "lifuren/poetry/PoetryModel.hpp"

[[maybe_unused]] static void testLidu() {
    const std::string path = lifuren::config::CONFIG.tmp;
    lifuren::poetry::LiduModel model({
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
    model.save(lifuren::config::CONFIG.tmp, "lidu.pt");
}

LFR_TEST(
    testLidu();
);
