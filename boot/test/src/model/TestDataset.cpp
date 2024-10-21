#include "lifuren/Test.hpp"

#include <random>

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/ImageDataset.hpp"
#include "lifuren/PoetryDataset.hpp"
#include "lifuren/EmbeddingClient.hpp"

#include "spdlog/fmt/ostr.h"

#include "torch/torch.h"

LFR_FORMAT_LOG_STREAM(at::Tensor);

[[maybe_unused]] static void testRawDataset() {
    std::random_device device;
    std::mt19937 rand(device());
    std::normal_distribution<> nd(10, 2);
    std::vector<float> labels;
    std::vector<std::vector<float>> features;
    labels.reserve(100);
    features.reserve(100);
    for(int index = 0; index < 100; ++index) {
        labels.push_back(nd(rand));
        std::vector<float> feature(10);
        std::for_each(feature.begin(), feature.end(), [&](auto& v) {
            v = nd(rand);
        });
        features.push_back(feature);
    }
    lifuren::dataset::RawDataset dataset(labels, features);
    SPDLOG_DEBUG("数据数量：{}", dataset.size().value());
    auto&& [
        feature,
        label
    ] = dataset.get(0);
    SPDLOG_DEBUG("数据特征：\n{}", feature);
    SPDLOG_DEBUG("数据标签：\n{}", label);
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
    auto&& [
        feature,
        label
    ] = dataset.get(0);
    SPDLOG_DEBUG("文件特征：\n{}", feature);
    SPDLOG_DEBUG("文件标签：\n{}", label);
}

[[maybe_unused]] static void testLoadImageFileDataset() {
    auto data_loader = lifuren::dataset::loadImageFileDataset(
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
    SPDLOG_DEBUG("图片特征：\n{}", data_loader->begin()->data);
    SPDLOG_DEBUG("图片标签：\n{}", data_loader->begin()->target);
}

[[maybe_unused]] static void testLoadPoetryFileDataset() {
    // // auto client = lifuren::EmbeddingClient::getClient("ollama");
    // auto client = lifuren::EmbeddingClient::getClient("ChineseWordVectors");
    // auto loader = lifuren::dataset::loadPoetryFileDataset(5, lifuren::file::join({lifuren::config::CONFIG.tmp, "poetry"}).string(), client.get());
    // // auto loader = lifuren::dataset::loadPoetryFileDataset(5, lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "poetry", "data"}).string(), client.get());
    // float* features = new float[300 * 5] { 0.0F };
    // loader.batchGet(3, features, nullptr);
    // auto pos = std::find_if(features, features + 1500, [](const auto& v) { return v == 0.0F; });
    // size_t distance = std::distance(features, pos);
    // SPDLOG_DEBUG("POS = {}", distance);
    // std::copy(features, features + 1500, std::ostream_iterator<float>(std::cout, " "));
    // delete features;
    // features = nullptr;
}

[[maybe_unused]] static void testPoetryEmbeddingFile() {
    std::vector<std::vector<float>> ww{ { 1.0F, 2.0F, 3.0F, 4.0F } };
    std::ofstream out;
    out.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "embedding.model" }).string(), std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    lifuren::dataset::poetry::write(out, ww);
    out.close();
    std::ifstream in;
    in.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "embedding.model" }).string(), std::ios_base::in | std::ios_base::binary);
    std::vector<std::vector<float>> rr;
    rr.reserve(1);
    lifuren::dataset::poetry::read(in, rr);
    in.close();
    assert(rr == ww);
}

LFR_TEST(
    // testRawDataset();
    // testFileDataset();
    // testLoadImageFileDataset();
    // testLoadPoetryFileDataset();
    testPoetryEmbeddingFile();
);
