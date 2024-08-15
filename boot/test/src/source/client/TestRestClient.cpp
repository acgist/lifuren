#include "lifuren/Client.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

#include <thread>
#include <chrono>

[[maybe_unused]] static void testGet() {
    lifuren::RestClient client{ "http://192.168.8.228:11434" };
    auto response = client.get("/");
    if(response) {
        SPDLOG_DEBUG("GET : {}", response->body);
    }
}

[[maybe_unused]] static void testPostStream() {
    lifuren::RestClient client{ "http://localhost:8080" };
    auto response = client.postStream("/sse", R"(
    // lifuren::RestClient client{ "http://192.168.8.228:11434" };
    // auto response = client.postStream("/api/chat", R"(
        {
            "model" : "glm4",
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
    // testGet();
    // testPostStream();
    std::thread thread{ testPostStream };
    thread.detach();
    std::this_thread::sleep_for(std::chrono::seconds(30));
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
