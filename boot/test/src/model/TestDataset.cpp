#include "lifuren/Test.hpp"

#include <random>

#include "torch/torch.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/EmbeddingClient.hpp"
#include "lifuren/audio/AudioDataset.hpp"
#include "lifuren/image/ImageDataset.hpp"
#include "lifuren/video/VideoDataset.hpp"
#include "lifuren/poetry/PoetryDataset.hpp"

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
    lifuren::logTensor("RAW数量", dataset.size().value());
    auto [
        feature,
        label
    ] = std::move(dataset.get(0));
    lifuren::logTensor("RAW特征", feature.sizes());
    lifuren::logTensor("RAW标签", label.sizes());
}

[[maybe_unused]] static void testLoadAudioFileDataset() {
    auto loader = lifuren::audio::loadFileDatasetLoader(200, lifuren::file::join({lifuren::config::CONFIG.tmp, "audio", "train"}).string());
    lifuren::logTensor("音频特征", loader->begin()->data.sizes());
    lifuren::logTensor("音频标签", loader->begin()->target.sizes());
}

[[maybe_unused]] static void testLoadImageFileDataset() {
    auto loader = lifuren::image::loadFileDatasetLoader(
        200,
        200,
        5,
        lifuren::file::join({lifuren::config::CONFIG.tmp, "gender", "train"}).string(),
        {
            { "man"  , 1.0F },
            { "woman", 0.0F }
        }
    );
    lifuren::logTensor("图片特征", loader->begin()->data.sizes());
    lifuren::logTensor("图片标签", loader->begin()->target.sizes());
}

[[maybe_unused]] static void testLoadVideoFileDataset() {
    auto loader = lifuren::video::loadFileDatasetLoader(640, 640, 200, lifuren::file::join({lifuren::config::CONFIG.tmp, "video", "train"}).string());
    lifuren::logTensor("视频特征", loader->begin()->data.sizes());
    lifuren::logTensor("视频标签", loader->begin()->target.sizes());
}

[[maybe_unused]] static void testLoadPoetryFileDataset() {
    auto loader = lifuren::poetry::loadFileDatasetLoader(5, lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "embedding.model"}).string());
    lifuren::logTensor("诗词特征", loader->begin()->data.sizes());
    lifuren::logTensor("诗词标签", loader->begin()->target.sizes());
}

LFR_TEST(
    testRawDataset();
    // testLoadAudioFileDataset();
    // testLoadImageFileDataset();
    // testLoadVideoFileDataset();
    // testLoadPoetryFileDataset();
);
