#include "./header/LibTorch.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    lifuren::Gender gener;
    // auto buffers = gener->named_buffers();
    // for (auto iterator = buffers.begin(); iterator != buffers.end(); ++iterator) {
    //     SPDLOG_DEBUG("buffers = {}", iterator->key());
    // }
    // auto parameters = gener->named_parameters();
    // for (auto iterator = parameters.begin(); iterator != parameters.end(); ++iterator) {
    //     SPDLOG_DEBUG("parameters = {}", iterator->key());
    // }
    lifuren::GenderHandler handler;
    handler.model = gener;
    handler.trainAndVal(
        1,
        32,
        0.01,
        #ifdef _WIN32
        "D:\\tmp\\gender",
        #else
        "/tmp/gender"
        #endif
        ".jpg",
        #ifdef _WIN32
        "D:\\tmp\\gender\\model.pt"
        #else
        "/tmp/gender/model.pt"
        #endif
    );
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
