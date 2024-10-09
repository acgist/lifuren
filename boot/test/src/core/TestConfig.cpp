#include "lifuren/Test.hpp"

#include <sstream>

#include "yaml-cpp/yaml.h"

#include "lifuren/Config.hpp"
#include "lifuren/Strings.hpp"
#include "lifuren/Poetrys.hpp"

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
    const std::string title   = "渔歌子·荻花秋";
    const std::string alias   = "渔父、渔父乐、渔父词";
    const std::string rhythm  = "渔歌子";
    const std::string content = lifuren::strings::trim(R"(
荻花秋，潇湘夜，橘洲佳景如屏画。
碧烟中，明月下，小艇垂纶初罢。
水为乡，篷作舍，鱼羹稻饭常餐也。
酒盈杯，书满架，名利不将心挂。
    )"s);
    int fontSize    = 0;
    int segmentSize = 0;
    std::vector<int> segmentRule;
    std::vector<std::string> aliasVector   = lifuren::strings::split(alias, lifuren::poetrys::POETRY_SIMPLE);
    std::vector<std::string> contentVector = lifuren::strings::split(content, lifuren::poetrys::POETRY_SIMPLE);
    std::for_each(contentVector.begin(), contentVector.end(), [&fontSize, &segmentSize, &segmentRule](auto& segment) {
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
)", rhythm, rhythm, lifuren::strings::join(aliasVector, ", "), title, content, fontSize, segmentSize, lifuren::strings::join(segmentRule, ", "));
  SPDLOG_DEBUG("配置格律：\n\n{}\n", config);
}

LFR_TEST(
    // testConfig();
    testRhythm();
    testGeneratePoetryConfig();
);
