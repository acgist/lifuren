#include "lifuren/Test.hpp"

#include <algorithm>

#include "lifuren/File.hpp"

[[maybe_unused]] static void testJoin() {
    SPDLOG_DEBUG("有效文件：{}", lifuren::file::join({       }).string());
    SPDLOG_DEBUG("有效文件：{}", lifuren::file::join({ "1"   }).string());
    SPDLOG_DEBUG("有效文件：{}", lifuren::file::join({ "./l" }).string());
    SPDLOG_DEBUG("有效文件：{}", lifuren::file::join({ "/", "/", "1" }).string());
    SPDLOG_DEBUG("有效文件：{}", lifuren::file::join({ "/", "2", "1" }).string());
    SPDLOG_DEBUG("有效文件：{}", lifuren::file::join({ "3", "2", "1" }).string());
    SPDLOG_DEBUG("文件后缀：{}", lifuren::file::fileSuffix("/path/filename"));
    SPDLOG_DEBUG("文件后缀：{}", lifuren::file::fileSuffix("/path/filename.m.zip"));
}

[[maybe_unused]] static void testListFile() {
    std::vector<std::string> vector;
    // lifuren::file::listFile(vector, lifuren::config::CONFIG.tmp);
    lifuren::file::listFile(vector, lifuren::config::CONFIG.tmp, { ".png", ".jpg" });
    std::for_each(vector.begin(), vector.end(), [](const std::string& path) {
        SPDLOG_DEBUG("有效文件：{}", path);
    });
}

[[maybe_unused]] static void testLoadFile() {
    const std::string content = lifuren::file::loadFile(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren.txt"}).string());
    SPDLOG_DEBUG("文件内容读取：{}", content);
}

[[maybe_unused]] static void testSaveFile() {
    const bool success = lifuren::file::saveFile(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren.txt"}).string(), "测试");
    SPDLOG_DEBUG("文件内容写出：{}", success);
}

LFR_TEST(
    testJoin();
    // testListFile();
    // testLoadFile();
    // testSaveFile();
);
