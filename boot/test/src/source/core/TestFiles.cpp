#include "Test.hpp"

#include <algorithm>

#include "lifuren/Files.hpp"

static void testListFiles() {
    std::vector<std::string> vector;
    // lifuren::files::listFiles(vector, "D:\\tmp");
    lifuren::files::listFiles(vector, "D:\\tmp", { ".png", ".jpg" });
    std::for_each(vector.begin(), vector.end(), [](const std::string& path) {
        SPDLOG_DEBUG("有效文件：{}", path);
    });
}

static void testLoadFile() {
    const std::string content = lifuren::files::loadFile("D:\\tmp\\lifuren.txt");
    SPDLOG_DEBUG("文件内容读取：{}", content);
}

static void testSaveFile() {
    const bool success = lifuren::files::saveFile("D:\\tmp\\lifuren.txt", "测试");
    SPDLOG_DEBUG("文件内容写出：{}", success);
}

LFR_TEST(
    testListFiles();
    testLoadFile();
    testSaveFile();
);
