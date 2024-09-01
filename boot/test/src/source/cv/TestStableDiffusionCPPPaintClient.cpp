#include "lifuren/Client.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testSD() {
    lifuren::StableDiffusionCPPPaintClient client{};
    client.paint({
        .prompt = "flower"
    }, [](bool finish, float percent, const std::string& message) {
        return true;
    });
}

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testSD();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
