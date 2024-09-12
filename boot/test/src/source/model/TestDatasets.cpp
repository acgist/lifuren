#include "Test.hpp"

#include <random>

#include "opencv2/opencv.hpp"

#include "lifuren/Datasets.hpp"

static void testReadImage() {
    float* data = new float[256 * 256 * 3];
    size_t length{ 0 };
    lifuren::datasets::readImage("D:/tmp/logo.png", data, length);
    cv::Mat image(256, 256, CV_8UC3);
    std::copy(data, data + length, image.data);
    cv::imwrite("D:/tmp/logo_copy.png", image);
    delete data;
    data = nullptr;
}

static void testRawDataset() {
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
    dataset.batchGet(1, f, l);
}

static void testFileDataset() {
    lifuren::datasets::FileDataset dataa(
        5,
        "D:/tmp",
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
        "D:/tmp/sex",
        { ".jpg" },
        [](const std::string& path, std::vector<std::vector<float>>& features) {
            SPDLOG_DEBUG("读取文件：{}", path);
            features.push_back(std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
        },
        {
            { "man"  , 1 },
            { "woman", 0 }
        }
    );
    size = datab.batchGet(0, f, l);
    SPDLOG_DEBUG("当前数量：{}", size);
    size = datab.batchGet(1, f, l);
    SPDLOG_DEBUG("当前数量：{}", size);
}

static void testShardingDataset() {
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

static void testLoadImageFileDataset() {
    std::map<std::string, int> mapping = {
        { "man"  , 1 },
        { "woman", 0 }
    };
    // auto data_loader = lifuren::datasets::loadImageFileDataset(200, 200, 20, "D:\\tmp\\gender\\train", ".jpg", mapping);
}

LFR_TEST(
    // testReadImage();
    // testRawDataset();
    testFileDataset();
    // testShardingDataset();
    // testLoadImageFileDataset();
);
