#include "Logger.hpp"
#include "utils/Datasets.hpp"

LFR_LOG_FORMAT_STREAM(at::Tensor);

static void testLoadImageFileDataset() {
    std::map<std::string, int> mapping = {
        { "man"  , 1 },
        { "woman", 0 }
    };
    auto data_loader = lifuren::datasets::loadImageFileDataset(200, 200, 20, "D:\\tmp\\gender\\train", ".jpg", mapping);
    auto data = data_loader.get();
    for(auto iterator = data->begin(); iterator != data->end(); ++iterator) {
        SPDLOG_DEBUG("数据：{}", iterator->target);
    }
}

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testLoadImageFileDataset();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
