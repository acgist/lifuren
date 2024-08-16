#include "lifuren/Lifuren.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    // yyyyMMddHHmmssxxxx
    SPDLOG_DEBUG("uuid = {}", lifuren::uuid());
    SPDLOG_DEBUG("uuid = {}", lifuren::uuid());
    SPDLOG_DEBUG("uuid = {}", lifuren::uuid());
    SPDLOG_DEBUG("uuid = {}", lifuren::uuid());
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
