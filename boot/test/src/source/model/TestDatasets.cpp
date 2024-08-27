#include "lifuren/Datasets.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

#include "spdlog/fmt/ostr.h"
#include "spdlog/fmt/chrono.h"
#include "spdlog/fmt/ranges.h"

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

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testLoadImageFileDataset();
    testDatasetSharding();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
