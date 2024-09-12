#include "Test.hpp"

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

static void testLoadImageFileDataset() {
    std::map<std::string, int> mapping = {
        { "man"  , 1 },
        { "woman", 0 }
    };
    auto data_loader = lifuren::datasets::loadImageFileDataset(200, 200, 20, "D:\\tmp\\gender\\train", ".jpg", mapping);
}

static void testDatasetSharding() {
    lifuren::datasets::DatasetSharding sharding;
    const size_t size = 100;
    sharding.type  = lifuren::datasets::Type::TRAIN;
    sharding.epoch = 0;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("训练数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 10));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 11));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 70));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 79));
    sharding.epoch = 1;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("训练数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 10));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 11));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 70));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 79));
    sharding.epoch = 8;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("训练数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 10));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 11));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 70));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 79));
    sharding.type  = lifuren::datasets::Type::VAL;
    sharding.epoch = 0;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("验证数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("验证数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("验证数据集索引: {}", sharding.getIndex(size, 1));
    sharding.epoch = 1;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("验证数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("验证数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("验证数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("验证数据集索引: {}", sharding.getIndex(size, 9));
    sharding.type  = lifuren::datasets::Type::TEST;
    sharding.epoch = 0;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("测试数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 1));
    sharding.epoch = 1;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("测试数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 9));
    sharding.trainRatio = 0.4F;
    sharding.valRatio   = 0.3F;
    sharding.testRatio  = 0.3F;
    sharding.type  = lifuren::datasets::Type::TRAIN;
    sharding.epoch = 0;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("训练数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 10));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 11));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 30));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 39));
    sharding.epoch = 1;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("训练数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 10));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 11));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 30));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 39));
    sharding.epoch = 4;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("训练数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 10));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 11));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 30));
    SPDLOG_DEBUG("训练数据集索引: {}", sharding.getIndex(size, 39));
    sharding.type  = lifuren::datasets::Type::VAL;
    sharding.epoch = 0;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("验证数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("验证数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("验证数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 9));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 29));
    sharding.epoch = 1;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("验证数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("验证数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("验证数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("验证数据集索引: {}", sharding.getIndex(size, 9));
    SPDLOG_DEBUG("验证数据集索引: {}", sharding.getIndex(size, 29));
    sharding.type  = lifuren::datasets::Type::TEST;
    sharding.epoch = 0;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("测试数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 9));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 29));
    sharding.epoch = 1;
    SPDLOG_DEBUG("================");
    SPDLOG_DEBUG("测试数据集大小: {}", sharding.getSize(size));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 0));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 1));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 9));
    SPDLOG_DEBUG("测试数据集索引: {}", sharding.getIndex(size, 29));
}

LFR_TEST(
    testReadImage();
    // testLoadImageFileDataset();
    // testDatasetSharding();
);
