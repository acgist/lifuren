#include "./header/LibTorch.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    std::vector<int> cfg_16bn = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    lifuren::Gender gener(cfg_16bn, 2);
    auto dict = gener->named_parameters();
    for (auto dictPtr = dict.begin(); dictPtr != dict.end(); dictPtr++) {
        std::cout << dictPtr->key() << std::endl;
        SPDLOG_DEBUG("dict = {}", dictPtr->key());
    }
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
