#include "lifuren/Files.hpp"

#include <algorithm>

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

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

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testListFiles();
    testLoadFile();
    testSaveFile();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
