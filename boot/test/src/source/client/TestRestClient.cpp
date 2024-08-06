#include "lifuren/Client.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testGet() {
    // lifuren::RestClient client{ "https://www.acgist.com" };
    lifuren::RestClient client{ "http://192.168.8.228:11434" };
    auto response = client.get("/");
    SPDLOG_DEBUG("GET : {}", response->body);
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testGet();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}