#include "lifuren/Test.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Image.hpp"
#include "lifuren/ImageModel.hpp"

[[maybe_unused]] static void testTrain() {
    const std::string path = lifuren::config::CONFIG.tmp;
    lifuren::image::WudaoziModel model({
        .lr         = 0.001F,
        .batch_size = 100,
        .epoch_size = 4,
        .model_name = "wudaozi",
        .train_path = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_TRAIN}).string(),
        .val_path   = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_VAL  }).string(),
        .test_path  = lifuren::file::join({path, "wudaozi", lifuren::config::DATASET_TEST }).string(),
    });
    model.define();
    model.trainValAndTest();
    model.save(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi.pt"}).string());
}

[[maybe_unused]] static void testPred() {
    auto client = lifuren::image::getImageClient("wudaozi");
    client->load(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi.pt"}).string());
    auto [success, output] = client->pred(lifuren::file::join({lifuren::config::CONFIG.tmp, "wudaozi.jpg"}).string());
    SPDLOG_INFO("输出结果：{} - {}", success, output);
}

LFR_TEST(
    // testTrain();
    testPred();
);
