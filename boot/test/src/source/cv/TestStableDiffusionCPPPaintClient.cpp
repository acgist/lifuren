#include "lifuren/PaintClient.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testSD(const std::string& prompt, const std::string& image, const std::string& output, const std::string& model) {
    lifuren::StableDiffusionCPPPaintClient client{};
    client.paint({
        .mode   = image.empty() ? lifuren::PaintClient::Mode::TXT2IMG : lifuren::PaintClient::Mode::IMG2IMG,
        .image  = image,
        .model  = model,
        .prompt = prompt,
        .output = output
    }, [](bool finish, float percent, const std::string& message) {
        return true;
    });
}

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    std::string model  = argc > 4 ? argv[4] : "";
    std::string output = argc > 3 ? argv[3] : "";
    std::string image  = argc > 2 ? argv[2] : "";
    std::string prompt = argc > 1 ? argv[1] : "flower";
    testSD(prompt, image, output, model);
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
