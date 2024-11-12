#include "lifuren/Test.hpp"

#include <random>

#include "torch/torch.h"

#include "spdlog/fmt/ostr.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/EmbeddingClient.hpp"
#include "lifuren/image/ImageDataset.hpp"
#include "lifuren/poetry/PoetryDataset.hpp"

LFR_FORMAT_LOG_STREAM(at::Tensor);
LFR_FORMAT_LOG_STREAM(c10::IntArrayRef)

[[maybe_unused]] static void testCsvDataset() {
    auto loader = lifuren::dataset::loadCsvDataset(5, lifuren::file::join({lifuren::config::CONFIG.tmp, "house", "train.csv"}).string());
    SPDLOG_DEBUG("CSV特征：\n{}", loader->begin()->data.sizes());
    SPDLOG_DEBUG("CSV标签：\n{}", loader->begin()->target.sizes());
    std::vector<torch::Tensor> features;
    lifuren::dataset::CsvDataset::loadCSV(lifuren::file::join({lifuren::config::CONFIG.tmp, "house", "test.csv"}).string(), features);
    SPDLOG_DEBUG("CSV特征：\n{}", features[0].sizes());
    SPDLOG_DEBUG("CSV特征：\n{}", features[1].sizes());
}

[[maybe_unused]] static void testRawDataset() {
    std::random_device device;
    std::mt19937 rand(device());
    std::normal_distribution<> nd(10, 2);
    std::vector<torch::Tensor> labels;
    std::vector<torch::Tensor> features;
    labels.reserve(100);
    features.reserve(100);
    for(int index = 0; index < 100; ++index) {
        labels.push_back(torch::tensor({ nd(rand) }));
        std::vector<float> feature(10);
        std::for_each(feature.begin(), feature.end(), [&](auto& v) {
            v = nd(rand);
        });
        features.push_back(torch::from_blob(feature.data(), { 10 }, torch::kFloat32).clone());
    }
    lifuren::dataset::RawDataset dataset(labels, features);
    SPDLOG_DEBUG("RAW数量：{}", dataset.size().value());
    auto [
        feature,
        label
    ] = std::move(dataset.get(0));
    SPDLOG_DEBUG("RAW特征：\n{}", feature.sizes());
    SPDLOG_DEBUG("RAW标签：\n{}", label.sizes());
}

[[maybe_unused]] static void testFileDataset() {
    lifuren::dataset::FileDataset dataset(
        lifuren::file::join({ lifuren::config::CONFIG.tmp, "gender", "train" }).string(),
        { ".jpg" },
        {
            { "man",   1.0F },
            { "woman", 0.0F }
        },
        [](const std::string& path) {
            // SPDLOG_DEBUG("读取文件：{}", path);
            return torch::rand({ 2, 2});
        }
    );
    SPDLOG_DEBUG("文件数量：{}", dataset.size().value());
    auto [
        feature,
        label
    ] = std::move(dataset.get(0));
    SPDLOG_DEBUG("文件特征：\n{}", feature.sizes());
    SPDLOG_DEBUG("文件标签：\n{}", label.sizes());
}

[[maybe_unused]] static void testLoadImageFileDataset() {
    auto loader = lifuren::dataset::loadImageFileDataset(
        200,
        200,
        5,
        lifuren::file::join({lifuren::config::CONFIG.tmp, "gender", "train"}).string(),
        ".jpg",
        {
            { "man"  , 1.0F },
            { "woman", 0.0F }
        }
    );
    SPDLOG_DEBUG("图片特征：\n{}", loader->begin()->data.sizes());
    SPDLOG_DEBUG("图片标签：\n{}", loader->begin()->target.sizes());
}

[[maybe_unused]] static void testLoadPoetryFileDataset() {
    auto loader = lifuren::dataset::loadPoetryFileDataset(5, lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "embedding.model"}).string());
    SPDLOG_DEBUG("诗词特征：\n{}", loader->begin()->data.sizes());
    SPDLOG_DEBUG("诗词标签：\n{}", loader->begin()->target.sizes());
}

LFR_TEST(
    // testCsvDataset();
    // testRawDataset();
    // testFileDataset();
    // testLoadImageFileDataset();
    testLoadPoetryFileDataset();
);
