#include "lifuren/Test.hpp"

#include "lifuren/PaintClient.hpp"

[[maybe_unused]] static void testSD(const std::string& prompt, const std::string& image, const std::string& output, const std::string& model) {
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

// ./testStableDiffusionCPPPaintClient flower /data/lifuren/ /data/stable-diffusion.cpp/build/bin/v1-5-pruned-emaonly.ckpt image.png

LFR_TEST(
    std::string image  = argc > 4 ? argv[4] : "";
    std::string model  = argc > 3 ? argv[3] : "";
    std::string output = argc > 2 ? argv[2] : "";
    std::string prompt = argc > 1 ? argv[1] : "flower";
    testSD(prompt, image, output, model);
);
