#include "lifuren/Test.hpp"

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
    const std::string title   = "登高";
    const std::string rhythm  = "七言律诗";
    const std::string content = lifuren::strings::trim(R"(
风急天高猿啸哀，渚清沙白鸟飞回。
无边落木萧萧下，不尽长江滚滚来。
万里悲秋常作客，百年多病独登台。
艰难苦恨繁霜鬓，潦倒新停浊酒杯。
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
    node["rhythm"]   = rhythm;
    node["alias"]    = std::vector<std::string>{};
    node["title"]    = title;
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
)", rhythm, rhythm, title, content, fontSize, segmentSize, lifuren::strings::join(segmentRule, ", "));
  SPDLOG_DEBUG("配置格律：\n\n{}\n", config);
}

LFR_TEST(
    // testConfig();
    testGeneratePoetryConfig();
);
