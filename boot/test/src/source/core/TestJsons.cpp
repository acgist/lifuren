#include "Test.hpp"

#include "lifuren/Jsons.hpp"

static void testLoadFile() {
    const nlohmann::json json = lifuren::jsons::loadFile<nlohmann::json>("D:\\tmp\\lifuren.json");
    SPDLOG_DEBUG("文件内容读取：{}", json.dump());
}

static void testSaveFile() {
    const bool success = lifuren::jsons::saveFile("D:\\tmp\\lifuren.json", "{}");
    SPDLOG_DEBUG("文件内容写出：{}", success);
}

LFR_TEST(
    testLoadFile();
    testSaveFile();
);
