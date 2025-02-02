#include "lifuren/Test.hpp"

#include "lifuren/video/VideoModel.hpp"

[[maybe_unused]] static void testWudaozi() {
    const std::string path = lifuren::config::CONFIG.tmp;
    lifuren::video::WudaoziModel model({
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
    model.save(lifuren::config::CONFIG.tmp, "wudaozi.pt");
}

LFR_TEST(
    testWudaozi();
);
