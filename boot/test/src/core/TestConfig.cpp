#include "lifuren/Test.hpp"

#include <numeric>
#include <sstream>

#include "yaml-cpp/yaml.h"

#include "lifuren/Config.hpp"
#include "lifuren/String.hpp"
#include "lifuren/poetry/Poetry.hpp"
#include "lifuren/poetry/PoetryDataset.hpp"

[[maybe_unused]] static void testUuid() {
    SPDLOG_DEBUG("uuid : {}", lifuren::config::uuid());
    SPDLOG_DEBUG("uuid : {}", lifuren::config::uuid());
}

[[maybe_unused]] static void testConfig() {
    std::stringstream stream;
    stream << lifuren::config::CONFIG.toYaml();
    SPDLOG_DEBUG("配置：\n{}", stream.str());
}

[[maybe_unused]] static void testRhythm() {
    for(const auto& [k, v] : lifuren::config::RHYTHM) {
        auto sc = std::accumulate(v.segmentRule.begin(),    v.segmentRule.end(),    0);
        auto pc = std::accumulate(v.participleRule.begin(), v.participleRule.end(), 0);
        SPDLOG_DEBUG("{} {} {} {}", v.title, v.fontSize, sc, pc);
        assert(sc == v.fontSize);
        assert(pc == v.fontSize);
    }
}

[[maybe_unused]] static void testRhythmConfig() {
    using namespace std::literals;
    const std::string title   = "醉垂鞭·双蝶绣罗裙";
    const std::string alias   = "";
    const std::string rhythm  = "醉垂鞭";
    const std::string content = lifuren::string::trim(R"(
双蝶绣罗裙，东池宴，初相见。
朱粉不深匀，闲花淡淡春。
细看诸处好，人人道，柳腰身。
昨日乱山昏，来时衣上云。
    )"s);
    int fontSize    = 0;
    int segmentSize = 0;
    std::vector<int> segmentRule;
    std::vector<std::string> aliasVector   = lifuren::string::split(alias,   lifuren::poetry::POETRY_SIMPLE);
    std::vector<std::string> contentVector = lifuren::string::split(content, lifuren::poetry::POETRY_SIMPLE);
    std::for_each(contentVector.begin(), contentVector.end(), [&fontSize, &segmentSize, &segmentRule](auto& segment) {
        segment = lifuren::string::trim(segment);
        if(segment.empty()) {
            return;
        }
        SPDLOG_DEBUG("处理诗句：{}", segment);
        const int length = lifuren::string::length(segment);
        fontSize += length;
        segmentSize++;
        segmentRule.push_back(length);
    });
    SPDLOG_DEBUG("诗句字数：{}", fontSize);
    SPDLOG_DEBUG("诗句段数：{}", segmentSize);
    SPDLOG_DEBUG("逐句字数：{}", lifuren::string::join(segmentRule, ","));
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
    // testUuid();
    // testConfig();
    // testRhythm();
    testRhythmConfig();
);
