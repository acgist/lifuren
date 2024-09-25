#include "Test.hpp"

#include "lifuren/Jsons.hpp"

[[maybe_unused]] static void testJSON() {
    nlohmann::json json = R"(
        {}
    )";
    // auto data = json["data"];
    // auto data = json.at("data");
    auto data = json.find("data");
    assert(data == json.end());
}

[[maybe_unused]] static void testLoadFile() {
    const nlohmann::json json = lifuren::jsons::loadFile<nlohmann::json>(lifuren::files::join({lifuren::config::CONFIG.tmp, "lifuren.json"}).string());
    SPDLOG_DEBUG("文件内容读取：{}", json.dump());
}

[[maybe_unused]] static void testSaveFile() {
    const bool success = lifuren::jsons::saveFile(lifuren::files::join({lifuren::config::CONFIG.tmp, "lifuren.json"}).string(), "{}");
    SPDLOG_DEBUG("文件内容写出：{}", success);
}

LFR_TEST(
    testJSON();
    // testLoadFile();
    // testSaveFile();
);
