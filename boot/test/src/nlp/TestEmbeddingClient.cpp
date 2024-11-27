#include "lifuren/Test.hpp"

#include "lifuren/File.hpp"
#include "lifuren/EmbeddingClient.hpp"

[[maybe_unused]] static void testOllamaEmbedding() {
    lifuren::OllamaEmbeddingClient client{};
    const auto v = std::move(client.getVector("李夫人"));
    SPDLOG_DEBUG("v length = {}", v.size());
}

[[maybe_unused]] static void testPepperEmbedding() {
    // lifuren::PepperEmbeddingClient ref{};
    {
        lifuren::PepperEmbeddingClient client{};
        auto v = std::move(client.getVector("东风"));
        // auto v = std::move(client.getVector({ "李", "夫", "人"}));
        SPDLOG_DEBUG("v length = {}", v.size());
    }
    SPDLOG_DEBUG("释放1");
    SPDLOG_DEBUG("释放2");
    SPDLOG_DEBUG("释放3");
}

[[maybe_unused]] static void testPepperEmbeddingFile() {
    auto client = std::make_unique<lifuren::PepperEmbeddingClient>();
    auto path = lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "ci" }).string();
    // auto path = lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "shi" }).string();
    // auto path = lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "songshi" }).string();
    // auto path = lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "tangshi" }).string();
    client->embedding(path);
}

LFR_TEST(
    // testOllamaEmbedding();
    // testPepperEmbedding();
    testPepperEmbeddingFile();
);
