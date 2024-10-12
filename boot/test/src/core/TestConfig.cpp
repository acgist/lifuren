#include "lifuren/Test.hpp"

#include <sstream>

#include "yaml-cpp/yaml.h"

#include "lifuren/Config.hpp"
#include "lifuren/String.hpp"
#include "lifuren/Poetry.hpp"

[[maybe_unused]] static void testConfig() {
    auto& config = lifuren::config::CONFIG;
    std::stringstream stream;
    stream << config.toYaml();
    SPDLOG_DEBUG("配置：{}", stream.str());
}

[[maybe_unused]] static void testRhythm() {
  for(const auto& [k, v] : lifuren::config::RHYTHM) {
    auto sc = std::accumulate(v.segmentRule.begin(), v.segmentRule.end(), 0);
    auto pc = std::accumulate(v.participleRule.begin(), v.participleRule.end(), 0);
    assert(sc == v.fontSize);
    assert(pc == v.fontSize);
  }
}

[[maybe_unused]] static void testGeneratePoetryConfig() {
    using namespace std::literals;
    const std::string title   = "满江红·无利无名";
    const std::string alias   = "上江虹、念良游、伤春曲";
    const std::string rhythm  = "满江红";
    const std::string content = lifuren::string::trim(R"(
无利无名，无荣无辱，无烦无恼。
夜灯前、独歌独酌，独吟独笑。
况值群山初雪满，又兼明月交光好。
便假饶百岁拟如何，从他老。
知富贵，谁能保。知功业，何时了。
算箪瓢金玉，所争多少。
一瞬光阴何足道，便思行乐常不早。
待春来携酒殢东风，眠芳草。
    )"s);
    int fontSize    = 0;
    int segmentSize = 0;
    std::vector<int> segmentRule;
    std::vector<std::string> aliasVector   = lifuren::string::split(alias, lifuren::poetry::POETRY_SIMPLE);
    std::vector<std::string> contentVector = lifuren::string::split(content, lifuren::poetry::POETRY_SIMPLE);
    std::for_each(contentVector.begin(), contentVector.end(), [&fontSize, &segmentSize, &segmentRule](auto& segment) {
        segment = lifuren::string::trim(segment);
        if(segment.empty()) {
            return;
        }
        SPDLOG_DEBUG("诗句：{}", segment);
        const int length = lifuren::string::length(segment);
        fontSize += length;
        segmentSize++;
        segmentRule.push_back(length);
    });
    YAML::Node node;
    SPDLOG_DEBUG("诗句字数：{}", fontSize);
    SPDLOG_DEBUG("诗句段数：{}", segmentSize);
    SPDLOG_DEBUG("逐句字数：{}", lifuren::string::join(segmentRule, ","));
    node["rhythm"]   = rhythm;
    node["alias"]    = aliasVector;
    node["title"]    = title;
    node["example"]  = content;
    node["fontSize"] = fontSize;
    node["segmentSize"] = segmentSize;
    node["segmentRule"] = segmentRule;
    node["participleRule"] = std::vector<int>{};
    std::stringstream stream;
    stream << node;
    SPDLOG_DEBUG("配置格律：\n{}", stream.str());
    std::transform(aliasVector.begin(), aliasVector.end(), aliasVector.begin(), [](const auto& v) {
      return "\"" + v + "\"";
    });
    std::string config = fmt::format(R"(
{:<s}:
  rhythm: {:<s}
  alias: [ {:<s} ]
  title: {:<s}
  example: |
    {:<s}
  fontSize: {:<d}
  segmentSize: {:<d}
  segmentRule: [ {:<s} ]
  participleRule: [
  ]
)", rhythm, rhythm, lifuren::string::join(aliasVector, ", "), title, content, fontSize, segmentSize, lifuren::string::join(segmentRule, ", "));
  SPDLOG_DEBUG("配置格律：\n\n{}\n", config);
}

LFR_TEST(
    // testConfig();
    testRhythm();
    testGeneratePoetryConfig();
);
