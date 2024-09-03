#include "lifuren/Client.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testSD(const std::string& image, const std::string& prompt) {
    lifuren::StableDiffusionCPPPaintClient client{};
    client.paint({
        .image  = image,
        .prompt = prompt
    }, [](bool finish, float percent, const std::string& message) {
        return true;
    });
}

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    std::string image  = argc > 1 ? argc[1] : "";
    std::string prompt = argc > 0 ? argv[0] : "flower";
    testSD(image, prompt);
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
