#include "lifuren/Client.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testGet() {
    // lifuren::RestClient client{ "https://www.acgist.com" };
    lifuren::RestClient client{ "http://192.168.8.228:11434" };
    auto response = client.get("/");
    SPDLOG_DEBUG("GET : {}", response->body);
}

static void testPostStream() {
    lifuren::RestClient client{ "http://192.168.8.228:11434" };
    auto response = client.postStream("/api/generate", R"(
        {
            "model" : "llama3.1",
            "prompt": "你好"
        }
    )", [](const char* data, size_t data_length) {
        SPDLOG_DEBUG("POST Stream callback : {}", std::string(data, data_length));
        return true;
    });
    SPDLOG_DEBUG("POST Stream : {}", response);
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    try {
        testGet();
        testPostStream();
    } catch(const std::exception& e) {
        SPDLOG_ERROR("Rest异常：{}", e.what());
    }
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}