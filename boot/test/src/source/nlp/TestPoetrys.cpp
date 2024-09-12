#include "Test.hpp"

#include "lifuren/Poetrys.hpp"
#include "lifuren/EmbeddingClient.hpp"

static void testPoetrys() {
    const std::string& poetry = R"(
    春花秋月何时了，往事知多少？小楼昨夜又东风，故国不堪回首月明中。
    雕栏玉砌应犹在，只是朱颜改。问君能有几多愁？恰似一江春水向东流。
    )";
    for(const auto& v : lifuren::poetrys::toChars(poetry)) {
        SPDLOG_DEBUG("char = {}", v);
    }
    for(const auto& v : lifuren::poetrys::toWords(poetry)) {
        SPDLOG_DEBUG("word = {}", v);
    }
    for(const auto& v : lifuren::poetrys::toSegments(poetry)) {
        SPDLOG_DEBUG("segment = {}", v);
    }
    SPDLOG_DEBUG("symbol = {}", lifuren::poetrys::replaceSymbol(poetry));
}

LFR_TEST(
    testPoetrys();
);
