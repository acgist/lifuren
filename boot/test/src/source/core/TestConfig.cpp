#include "Test.hpp"

#include <sstream>

#include "yaml-cpp/yaml.h"

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
    const std::string rhythm  = "七言绝句";
    const std::string content = lifuren::strings::trim(R"(
月落乌啼霜满天，江枫渔火对愁眠。
姑苏城外寒山寺，夜半钟声到客船。
    )"s);
    int fontSize    = 0;
    int segmentSize = 0;
    std::vector<int> segmentRule;
    std::vector<std::string> vector = lifuren::strings::split(content, std::vector<std::string>{ "，", "。", "？", "！" });
    std::for_each(vector.begin(), vector.end(), [&fontSize, &segmentSize, &segmentRule](auto& segment) {
        segment = lifuren::strings::trim(segment);
        if(segment.empty()) {
            return;
        }
        SPDLOG_DEBUG("诗句：{}", segment);
        const int length = lifuren::strings::length(segment);
        fontSize += length;
        segmentSize++;
        segmentRule.push_back(length);
    });
    YAML::Node node;
    SPDLOG_DEBUG("诗句字数：{}", fontSize);
    SPDLOG_DEBUG("诗句段数：{}", segmentSize);
    SPDLOG_DEBUG("逐句字数：{}", lifuren::strings::join(segmentRule, ","));
    node["rhythm"]   = "";
    node["alias"]    = std::vector<std::string>{};
    node["title"]    = "";
    node["example"]  = content;
    node["fontSize"] = fontSize;
    node["segmentSize"] = segmentSize;
    node["segmentRule"] = segmentRule;
    node["participleRule"] = std::vector<int>{};
    std::stringstream stream;
    stream << node;
    SPDLOG_DEBUG("配置格律：\n{}", stream.str());
    std::string config = fmt::format(R"(
{:<s}:
  rhythm: {:<s}
  alias: []
  title: {:<s}
  example: |
    {:<s}
  fontSize: {:<d}
  segmentSize: {:<d}
  segmentRule: [ {:<s} ]
  participleRule: [
  ]
)", rhythm, rhythm, rhythm, content, fontSize, segmentSize, lifuren::strings::join(segmentRule, ", "));
  SPDLOG_DEBUG("配置格律：\n\n{}\n", config);
}

LFR_TEST(
    // testConfig();
    testGeneratePoetryConfig();
);
