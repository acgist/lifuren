#include "lifuren/Client.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testSD(const std::string& image, const std::string& prompt, const std::string& output) {
    lifuren::StableDiffusionCPPPaintClient client{};
    client.paint({
        .image  = image,
        .prompt = prompt,
        .output = output
    }, [](bool finish, float percent, const std::string& message) {
        return true;
    });
}

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    std::string output = argc > 2 ? argv[2] : "";
    std::string image  = argc > 1 ? argv[1] : "";
    std::string prompt = argc > 0 ? argv[0] : "flower";
    testSD(image, prompt, output);
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
