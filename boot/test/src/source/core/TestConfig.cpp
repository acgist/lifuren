#include "Test.hpp"

#include <sstream>

#include "nlohmann/json.hpp"

#include "lifuren/Config.hpp"
#include "lifuren/Strings.hpp"

[[maybe_unused]] static void testConfig() {
    auto& config = lifuren::config::CONFIG;
    std::stringstream stream;
    stream << config.toYaml();
    SPDLOG_DEBUG("配置：{}", stream.str());
}

[[maybe_unused]] static void testGeneratePoetryConfig() {
    using namespace std::literals;
    const std::string content = lifuren::strings::trim(R"(
月落乌啼霜满天，江枫渔火对愁眠。
姑苏城外寒山寺，夜半钟声到客船。
    )"s);
    int fontSize    = 0;
    int segmentSize = 0;
    std::vector<int> segmentRule;
    nlohmann::json json;
    std::vector<std::string> vector = lifuren::strings::split(content, std::vector<std::string>{ "，", "。", "？", "！" });
    std::for_each(vector.begin(), vector.end(), [&fontSize, &segmentSize, &segmentRule](auto& segment) {
        segment = lifuren::strings::trim(segment);
        if(segment.empty()) {
            return;
        }
        SPDLOG_DEBUG("诗句：{}", segment);
        int length = lifuren::strings::length(segment);
        fontSize += length;
        segmentSize++;
        segmentRule.push_back(length);
    });
    SPDLOG_DEBUG("诗句字数：{}", fontSize);
    SPDLOG_DEBUG("诗句段数：{}", segmentSize);
    SPDLOG_DEBUG("逐句字数：{}", lifuren::strings::join(segmentRule, ","));
    json["example"]  = content;
    json["fontSize"] = fontSize;
    json["segmentSize"] = segmentSize;
    json["segmentRule"] = segmentRule;
    json["participleRule"] = std::vector<int>{};
    SPDLOG_DEBUG("配置规则：{}", json.dump(2));
}

LFR_TEST(
    testConfig();
    testGeneratePoetryConfig();
);
