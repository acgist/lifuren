#include "Test.hpp"

#include <random>

#include "opencv2/opencv.hpp"

#include "lifuren/Files.hpp"
#include "lifuren/Datasets.hpp"
#include "lifuren/ImageDatasets.hpp"
#include "lifuren/PoetryDatasets.hpp"
#include "lifuren/EmbeddingClient.hpp"

[[maybe_unused]] static void testRawDataset() {
    std::random_device device;
    std::mt19937 rand(device());
    std::normal_distribution<> nd(10, 2);
    float features[210] { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
    float labels  [210] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    for(int index = 10; index < 210; ++index) {
        features[index] = nd(rand);
        labels  [index] = nd(rand);
    }
    lifuren::datasets::RawDataset dataset(
        210LL,
        5,
        features,
        1,
        labels,
        1
    );
    assert(dataset.getCount()      == 210LL);
    assert(dataset.getBatchSize()  == 5LL);
    assert(dataset.getBatchCount() == 42LL);
    float f[5];
    float l[5];
    dataset.batchGet(0, f, l);
    std::copy(f, f + 5, std::ostream_iterator<float>(std::cout, " "));
    std::cout << '\n';
    std::copy(l, l + 5, std::ostream_iterator<float>(std::cout, " "));
    std::cout << '\n';
    dataset.batchGet(1, f, l);
    std::copy(f, f + 5, std::ostream_iterator<float>(std::cout, " "));
    std::cout << '\n';
    std::copy(l, l + 5, std::ostream_iterator<float>(std::cout, " "));
    std::cout << '\n';
}

[[maybe_unused]] static void testFileDataset() {
    lifuren::datasets::FileDataset dataa(
        5,
        lifuren::config::CONFIG.tmp,
        { ".jpg" },
        [](const std::string& path, std::vector<std::vector<float>>& features) {
            SPDLOG_DEBUG("读取文件：{}", path);
            features.push_back(std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
        }
    );
    float f[5];
    float l[5];
    size_t size = dataa.batchGet(0, f, l);
    SPDLOG_DEBUG("当前数量：{}", size);
    size = dataa.batchGet(1, f, l);
    SPDLOG_DEBUG("当前数量：{}", size);
    lifuren::datasets::FileDataset datab(
        5,
        lifuren::files::join({lifuren::config::CONFIG.tmp, "sex"}).string(),
        { ".jpg" },
        [](const std::string& path, std::vector<std::vector<float>>& features) {
            SPDLOG_DEBUG("读取文件：{}", path);
            features.push_back(std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
        },
        {
            { "man"  , 1.0F },
            { "woman", 0.0F }
        }
    );
    size = datab.batchGet(0, f, l);
    SPDLOG_DEBUG("当前数量：{}", size);
    size = datab.batchGet(1, f, l);
    SPDLOG_DEBUG("当前数量：{}", size);
}

[[maybe_unused]] static void testShardingDataset() {
    std::random_device device;
    std::mt19937 rand(device());
    std::normal_distribution<> nd(10, 2);
    float features[210] { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
    float labels  [210] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    for(int index = 10; index < 210; ++index) {
        features[index] = nd(rand);
        labels  [index] = nd(rand);
    }
    auto dataset = std::make_shared<lifuren::datasets::RawDataset>(
        210LL,
        5,
        features,
        1,
        labels,
        1
    );
    auto [traina, vala, testa] = lifuren::datasets::ShardingDataset::make(dataset);
    assert(traina.getCount()      == 205LL);
    assert(traina.getBatchSize()  == 5LL);
    assert(traina.getBatchCount() == 41LL);
    assert(vala.getCount()        == 5LL);
    assert(vala.getBatchSize()    == 5LL);
    assert(vala.getBatchCount()   == 1LL);
    assert(testa.getCount()       == 0LL);
    assert(testa.getBatchSize()   == 5LL);
    assert(testa.getBatchCount()  == 0LL);
    float f[5];
    float l[5];
    traina.batchGet(0, f, l);
    vala.batchGet(0, f, l);
    auto [trainb, valb, testb] = lifuren::datasets::ShardingDataset::make(dataset, 0, 2, 3, 1);
    assert(trainb.getCount()      == 195LL);
    assert(trainb.getBatchSize()  == 5LL);
    assert(trainb.getBatchCount() == 39LL);
    assert(valb.getCount()        == 10LL);
    assert(valb.getBatchSize()    == 5LL);
    assert(valb.getBatchCount()   == 2LL);
    assert(testb.getCount()       == 5LL);
    assert(testb.getBatchSize()   == 5LL);
    assert(testb.getBatchCount()  == 1LL);
}

[[maybe_unused]] static void testLoadImageFileDataset() {
    std::map<std::string, float> mapping {
        { "man"  , 1.0F },
        { "woman", 0.0F }
    };
    auto data_loader = lifuren::datasets::loadImageFileDataset(200, 200, 5, lifuren::files::join({lifuren::config::CONFIG.tmp, "sex"}).string(), ".jpg", mapping);
    float* features  = new float[5 * 200 * 200 * 3];
    float* labels    = new float[5];
    data_loader.batchGet(0, features, labels);
    uint8_t data[120000];
    std::copy(features, features + 120000, data);
    lifuren::images::write(lifuren::files::join({lifuren::config::CONFIG.tmp, "loader.png"}).string(), data, 200, 200);
    for(size_t i = 0; i < data_loader.getBatchCount(); ++i) {
        data_loader.batchGet(i, features, labels);
        std::copy(labels, labels + 5, std::ostream_iterator<float>(std::cout, " "));
        std::cout << '\n';
    }
    delete features;
    features = nullptr;
    delete labels;
    labels = nullptr;
}

[[maybe_unused]] static void testLoadPoetryFileDataset() {
    // auto client = lifuren::EmbeddingClient::getClient("ollama");
    auto client = lifuren::EmbeddingClient::getClient("ChineseWordVectors");
    auto loader = lifuren::loadPoetryFileDataset(5, lifuren::files::join({lifuren::config::CONFIG.tmp, "poetry"}).string(), client.get());
    // auto loader = lifuren::loadPoetryFileDataset(5, lifuren::files::join({lifuren::config::CONFIG.tmp, "lifuren", "poetry", "data"}).string(), client.get());
    float* features = new float[300 * 5] { 0 };
    loader.batchGet(3, features, nullptr);
    auto pos = std::find_if(features, features + 1500, [](const auto& v) { return v == 0.0F; });
    size_t distance = std::distance(features, pos);
    SPDLOG_DEBUG("POS = {}", distance);
    std::copy(features, features + 1500, std::ostream_iterator<float>(std::cout, " "));
    delete features;
    features = nullptr;
}

LFR_TEST(
    // testRawDataset();
    // testFileDataset();
    // testShardingDataset();
    // testLoadImageFileDataset();
    testLoadPoetryFileDataset();
);
