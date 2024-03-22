#include "utils/Files.hpp"

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    std::vector<std::string> vector;
    // lifuren::files::listFiles(vector, "D:\\tmp");
    lifuren::files::listFiles(vector, "D:\\tmp", { ".png", ".jpg" });
    std::for_each(vector.begin(), vector.end(), [](const std::string& path) {
        SPDLOG_DEBUG("有效文件：{}", path);
    });
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
