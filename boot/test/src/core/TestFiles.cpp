#include "lifuren/Test.hpp"

#include <algorithm>

#include "lifuren/Files.hpp"

[[maybe_unused]] static void testListFiles() {
    std::vector<std::string> vector;
    // lifuren::files::listFiles(vector, lifuren::config::CONFIG.tmp);
    lifuren::files::listFiles(vector, lifuren::config::CONFIG.tmp, { ".png", ".jpg" });
    std::for_each(vector.begin(), vector.end(), [](const std::string& path) {
        SPDLOG_DEBUG("有效文件：{}", path);
    });
}

[[maybe_unused]] static void testLoadFile() {
    const std::string content = lifuren::files::loadFile(lifuren::files::join({lifuren::config::CONFIG.tmp, "lifuren.txt"}).string());
    SPDLOG_DEBUG("文件内容读取：{}", content);
}

[[maybe_unused]] static void testSaveFile() {
    const bool success = lifuren::files::saveFile(lifuren::files::join({lifuren::config::CONFIG.tmp, "lifuren.txt"}).string(), "测试");
    SPDLOG_DEBUG("文件内容写出：{}", success);
}

LFR_TEST(
    testListFiles();
    testLoadFile();
    testSaveFile();
);
