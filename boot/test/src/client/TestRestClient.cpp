#include "lifuren/Test.hpp"

#include "lifuren/Client.hpp"

static lifuren::RestClient client{ "http://192.168.8.228:11434" };

[[maybe_unused]] static void testHead() {
    auto response = std::move(client.head("/"));
    if(response) {
        SPDLOG_DEBUG("HEAD : {}", response.success);
    }
}

[[maybe_unused]] static void testGet() {
    auto response = std::move(client.get("/"));
    if(response) {
        SPDLOG_DEBUG("GET : {}", response.body);
    }
}

[[maybe_unused]] static void testPostStream() {
    auto success = client.postStream("/api/generate", R"(
        {
            "model" : "glm4",
            "prompt": "你好"
        }
    )", [](const char* data, size_t data_length) {
        SPDLOG_DEBUG("POST Stream callback : {}", std::string(data, data_length));
        return true;
    });
    SPDLOG_DEBUG("POST Stream : {}", success);
}

[[maybe_unused]] static void testToQuery() {
    auto query = std::move(lifuren::http::toQuery({
        { "name",    "碧螺萧萧"    },
        { "profile", "acgist.png" }
    }));
    SPDLOG_DEBUG("query : {}", query);
}

LFR_TEST(
    testHead();
    testGet();
    testPostStream();
    // testToQuery();
);
