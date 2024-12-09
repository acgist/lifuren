#include "lifuren/Test.hpp"

#include <random>

#include "torch/torch.h"

#include "spdlog/fmt/ostr.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/EmbeddingClient.hpp"
#include "lifuren/audio/AudioDataset.hpp"
#include "lifuren/image/ImageDataset.hpp"
#include "lifuren/poetry/PoetryDataset.hpp"

LFR_FORMAT_LOG_STREAM(at::Tensor);
LFR_FORMAT_LOG_STREAM(c10::IntArrayRef)

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
    // lifuren::dataset::FileDataset dataset(
    //     lifuren::file::join({ lifuren::config::CONFIG.tmp, "gender", "train" }).string(),
    //     { ".jpg" },
    //     {
    //         { "man",   1.0F },
    //         { "woman", 0.0F }
    //     },
    //     [](const std::string& path) {
    //         return torch::rand({ 2, 2});
    //     }
    // );
    // lifuren::dataset::FileDataset dataset(
    //     lifuren::file::join({ lifuren::config::CONFIG.tmp, "dataset1", "file.txt" }).string(),
    //     [](const std::string& file, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& type) {
    //         labels.push_back(torch::rand({ 2, 2}));
    //         features.push_back(torch::rand({ 2, 2}));
    //     }
    // );
    // lifuren::dataset::FileDataset dataset(
    //     lifuren::file::join({ lifuren::config::CONFIG.tmp, "dataset2" }).string(),
    //     { ".txt", ".json" },
    //     [](const std::string& file, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& type) {
    //         labels.push_back(torch::rand({ 2, 2}));
    //         features.push_back(torch::rand({ 2, 2}));
    //     }
    // );
    // lifuren::dataset::FileDataset dataset(
    //     lifuren::file::join({ lifuren::config::CONFIG.tmp, "dataset3" }).string(),
    //     ".json",
    //     { ".txt" },
    //     [](const std::string& file, const std::string& label, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& type) {
    //         labels.push_back(torch::rand({ 2, 2}));
    //         features.push_back(torch::rand({ 2, 2}));
    //     }
    // );
    lifuren::dataset::FileDataset dataset(
        lifuren::file::join({ lifuren::config::CONFIG.tmp, "dataset4" }).string(),
        "source",
        "target",
        { ".txt" },
        [](const std::string& file, const std::string& label, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& type) {
            labels.push_back(torch::rand({ 2, 2}));
            features.push_back(torch::rand({ 2, 2}));
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

[[maybe_unused]] static void testEmbeddingSlice() {
    std::ifstream in;
    std::ofstream val;
    std::ofstream train;
    in.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "embedding.model" }).string(), std::ios_base::in | std::ios_base::binary);
    val.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "embedding_val.model" }).string(), std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    train.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "embedding_train.model" }).string(), std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    std::vector<std::vector<float>> rr;
    int i = 0;
    while(!lifuren::dataset::poetry::read(in, rr)) {
        if(++i % 100 == 0) {
            lifuren::dataset::poetry::write(val, rr);
        } else {
            lifuren::dataset::poetry::write(train, rr);
        }
        rr.clear();
    }
    lifuren::dataset::poetry::writeEnd(val, lifuren::dataset::poetry::END_OF_DATASET);
    lifuren::dataset::poetry::writeEnd(train, lifuren::dataset::poetry::END_OF_DATASET);
    in.close();
    val.close();
    train.close();
}

[[maybe_unused]] static void testStftIstft() {
    std::ifstream input;
    std::ofstream output;
    input .open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "noise.pcm" }).string(),   std::ios_base::binary);
    output.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "denoise.pcm" }).string(), std::ios_base::binary);
    int size = 48000;
    std::vector<short> data;
    data.resize(size);
    float norm_factor;
    while(input.read(reinterpret_cast<char*>(data.data()), size * sizeof(short))) {
        // 其他处理
        auto tuple = std::move(lifuren::dataset::audio::pcm_mag_pha_stft(data, norm_factor));
        auto pcm   = std::move(lifuren::dataset::audio::pcm_mag_pha_istft(std::get<0>(tuple), std::get<1>(tuple), norm_factor));
        // 原始函数
        // int n_fft    = 400;
        // int hop_size = 100;
        // int win_size = 400;
        // auto window = torch::hann_window(win_size);
        // auto pcm_tensor = torch::zeros({1, static_cast<int>(data.size())}, torch::kFloat32);
        // float* tensor_data = reinterpret_cast<float*>(pcm_tensor.data_ptr());
        // std::copy_n(data.data(), data.size(), tensor_data);
        // auto spec   = torch::stft(pcm_tensor, n_fft, hop_size, win_size, window, true, "reflect", false, std::nullopt, true);
        // auto result = torch::istft(spec, n_fft, hop_size, win_size, window, true);
        // tensor_data = reinterpret_cast<float*>(result.data_ptr());
        // std::vector<short> pcm;
        // pcm.resize(result.sizes()[1]);
        // std::copy_n(tensor_data, pcm.size(), pcm.data());
        output.write(reinterpret_cast<char*>(pcm.data()), pcm.size() * sizeof(short));
        output.flush();
    }
    input.close();
    output.close();
}

LFR_TEST(
    // testRawDataset();
    // testFileDataset();
    // testLoadImageFileDataset();
    // testLoadPoetryFileDataset();
    // testEmbeddingSlice();
    testStftIstft();
);
