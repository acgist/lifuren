#include "lifuren/Test.hpp"

#include "lifuren/audio/ComposeModel.hpp"

[[maybe_unused]] static void testShikuang() {
    const std::string path = lifuren::config::CONFIG.tmp;
    lifuren::ShikuangModel model({
        .model_name = "baicai",
        .train_path = lifuren::file::join({path, "baicai", lifuren::config::DATASET_TRAIN}).string(),
        .val_path   = lifuren::file::join({path, "baicai", lifuren::config::DATASET_VAL}).string(),
        .test_path  = lifuren::file::join({path, "baicai", lifuren::config::DATASET_TEST}).string(),
    });
    model.define();
    model.trainValAndTest();
    model.save(lifuren::config::CONFIG.tmp, "shikuang.pt");
}

LFR_TEST(
    testShikuang();
);
