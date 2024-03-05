#include <regex>
#include <string>
#include <vector>
#include <algorithm>

#include "Logger.hpp"
#include "Strings.hpp"
#include "Collections.hpp"

#include "nlohmann/json.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    const std::string content = lifuren::strings::trim(R"(
庭院深深深几许，杨柳堆烟，帘幕无重数。玉勒雕鞍游冶处，楼高不见章台路。
雨横风狂三月暮，门掩黄昏，无计留春住。泪眼问花花不语，乱红飞过秋千去。
    )");
    int fontSize = 0;
    int segmentSize = 0;
    std::vector<int> segmentFontSize;
    nlohmann::json json;
    std::vector<std::string> vector = lifuren::collections::split(content, std::vector<std::string>{ "，", "。", "？", "！" });
    std::for_each(vector.begin(), vector.end(), [&fontSize, &segmentSize, &segmentFontSize](auto& segment) {
        segment = lifuren::strings::trim(segment);
        if(segment.empty()) {
            return;
        }
        SPDLOG_DEBUG("诗句：{}", segment);
        int length = lifuren::strings::length(segment.c_str());
        fontSize += length;
        segmentSize++;
        segmentFontSize.push_back(length);
    });
    SPDLOG_DEBUG("诗句字数：{}", fontSize);
    SPDLOG_DEBUG("诗句段数：{}", segmentSize);
    SPDLOG_DEBUG("逐句字数：{}", lifuren::collections::join(segmentFontSize, ","));
    json["example"] = content;
    json["fontSize"] = fontSize;
    json["segmentSize"] = segmentSize;
    json["segmentRule"] = segmentFontSize;
    json["participleRule"] = std::vector<int>{ };
    SPDLOG_DEBUG("配置规则：{}", json.dump(2));
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
