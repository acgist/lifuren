#include "lifuren/Test.hpp"

#include <algorithm>

#include "lifuren/File.hpp"

[[maybe_unused]] static void testJoin() {
    SPDLOG_DEBUG("文件路径：{}", lifuren::file::join({       }).string());
    SPDLOG_DEBUG("文件路径：{}", lifuren::file::join({ "1"   }).string());
    SPDLOG_DEBUG("文件路径：{}", lifuren::file::join({ "./l" }).string());
    SPDLOG_DEBUG("文件路径：{}", lifuren::file::join({ "/", "/", "1" }).string());
    SPDLOG_DEBUG("文件路径：{}", lifuren::file::join({ "/", "2", "1" }).string());
    SPDLOG_DEBUG("文件路径：{}", lifuren::file::join({ "3", "2", "1" }).string());
}

[[maybe_unused]] static void testListFile() {
    std::vector<std::string> vector;
    // lifuren::file::list_file(vector, lifuren::config::CONFIG.tmp);
    lifuren::file::list_file(vector, lifuren::file::join({lifuren::config::CONFIG.tmp, "face"}).string(), { ".png", ".jpg" });
    std::for_each(vector.begin(), vector.end(), [](const std::string& path) {
        SPDLOG_DEBUG("有效文件：{}", path);
    });
}

[[maybe_unused]] static void testSuffix() {
    SPDLOG_DEBUG("文件后缀：{}", lifuren::file::file_suffix("/path/filename"));
    SPDLOG_DEBUG("文件后缀：{}", lifuren::file::file_suffix("/path/filename.m.zip"));
    auto filename = lifuren::file::modify_filename("/root/test.txt", ".exe");
    SPDLOG_DEBUG("新的文件名称：{}", filename);
    filename = lifuren::file::modify_filename("/root/test.txt", ".exe", "gen");
    SPDLOG_DEBUG("新的文件名称：{}", filename);
}

LFR_TEST(
    testJoin();
    testListFile();
    testSuffix();
);
