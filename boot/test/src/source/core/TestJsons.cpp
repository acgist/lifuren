#include "lifuren/Jsons.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testLoadFile() {
    const nlohmann::json json = lifuren::jsons::loadFile<nlohmann::json>("D:\\tmp\\lifuren.json");
    SPDLOG_DEBUG("文件内容读取：{}", json.dump());
}

static void testSaveFile() {
    const bool success = lifuren::jsons::saveFile("D:\\tmp\\lifuren.json", "{}");
    SPDLOG_DEBUG("文件内容写出：{}", success);
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testLoadFile();
    testSaveFile();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
