#include "Test.hpp"

#include <thread>
#include <chrono>

#include "lifuren/Client.hpp"

[[maybe_unused]] static void testGet() {
    lifuren::RestClient client{ "http://192.168.8.228:11434" };
    auto response = client.get("/");
    if(response) {
        SPDLOG_DEBUG("GET : {}", response.body);
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

LFR_TEST(
    // testGet();
    // testPostStream();
    std::thread thread{ testPostStream };
    thread.detach();
    std::this_thread::sleep_for(std::chrono::seconds(30));
);
