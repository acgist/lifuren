#include "lifuren/Test.hpp"

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Poetry.hpp"

[[maybe_unused]] static void testPoetry() {
    nlohmann::json json = nlohmann::json::parse(R"(
    {
        "author": "杜甫", 
        "paragraphs": [
            "国破山河在，城春草木深。",
            "感时花溅泪，恨别鸟惊心。",
            "烽火连三月，家书抵万金。",
            "白头掻更短，浑欲不胜簪。"
        ], 
        "rhythmic": "春望"
    }
    )");
    // nlohmann::json json = nlohmann::json::parse(R"(
    // {
    //     "author": "欧阳修", 
    //     "paragraphs": [
    //         "庭院深深深几许，杨柳堆烟，帘幕无重数。",
    //         "玉勒雕鞍游冶处，楼高不见章台路。",
    //         "雨横风狂三月暮，门掩黄昏，无计留春住。",
    //         "泪眼问花花不语，乱红飞过秋千去。"
    //     ], 
    //     "rhythmic": "蝶恋花"
    // }
    // )");
    // nlohmann::json json = nlohmann::json::parse(R"(
    // {
    //     "author": "朱敦儒", 
    //     "paragraphs": [
    //         "我是清都山水郎，天教懒慢带疏狂。", 
    //         "曾批给露支风敕，累奏留云借月章。", 
    //         "诗万首，酒千觞，几曾着眼看侯王？玉楼金阙慵归去，且插梅花醉洛阳。"
    //     ], 
    //     "rhythmic": "鹧鸪天"
    // }
    // )");
    lifuren::poetry::Poetry poetry = json;
    poetry.preproccess();
    SPDLOG_DEBUG("匹配格律：\n{}", poetry.matchRhythm());
    SPDLOG_DEBUG("原始段落：\n{}", poetry.segment);
    SPDLOG_DEBUG("朴素段落：\n{}", poetry.simpleSegment);
    if(poetry.matchRhythm()) {
        poetry.participle();
        SPDLOG_DEBUG("分词段落：\n{}", poetry.participleSegment);
    }
    lifuren::poetry::Poetry diff = json;
    assert(diff == poetry);
}

[[maybe_unused]] static void testMatchDataset() {
    std::vector<std::string> files;
    lifuren::file::listFile(files, lifuren::config::CONFIG.mark.begin()->path, { ".json" });
    int64_t fSize    = 0LL;
    int64_t count    = 0LL;
    int64_t total    = 0LL;
    int64_t ciCount  = 0LL;
    int64_t ciTotal  = 0LL;
    int64_t shiCount = 0LL;
    int64_t shiTotal = 0LL;
    std::map<std::string, int64_t> unciCount;
    std::map<std::string, int64_t> unshiCount;
    std::map<std::string, int64_t> rhythmCount;
    for(const auto& file : files) {
        ++fSize;
        std::string&& json = lifuren::file::loadFile(file);
        auto&& poetries = nlohmann::json::parse(json);
        for(const auto& poetry : poetries) {
            lifuren::poetry::Poetry value = poetry;
            const bool ci = poetry.contains("rhythmic");
            value.preproccess();
            ++total;
            if(ci) {
                ++ciTotal;
            } else {
                ++shiTotal;
            }
            if(value.matchRhythm()) {
                auto iter = rhythmCount.find(value.rhythmPtr->rhythm);
                if(iter == rhythmCount.end()) {
                    iter = rhythmCount.emplace(value.rhythmPtr->rhythm, 0).first;
                }
                ++iter->second;
                // SPDLOG_DEBUG("匹配成功：{} - {}", value.rhythmPtr->rhythm, value.simpleSegment);
                ++count;
                if(ci) {
                    ++ciCount;
                } else {
                    ++shiCount;
                }
            } else {
                if(poetry.contains("rhythmic")) {
                    auto rhythm = poetry.at("rhythmic").get<std::string>();
                    auto iter = unciCount.find(rhythm);
                    if(iter == unciCount.end()) {
                        iter = unciCount.emplace(rhythm, 0).first;
                    }
                    --iter->second;
                } else {
                    auto rhythm = poetry.at("title").get<std::string>();
                    auto iter = unshiCount.find(rhythm);
                    if(iter == unshiCount.end()) {
                        iter = unshiCount.emplace(rhythm, 0).first;
                    }
                    --iter->second;
                }
                // SPDLOG_DEBUG("匹配失败：{}", rhythm);
                // SPDLOG_DEBUG("匹配失败：{}", poetry.dump());
                // return;
            }
            if(total % 1000 == 0) {
                SPDLOG_DEBUG("当前数量：{} / {} / {}", fSize, count, total);
            }
        }
    }
    std::vector<std::pair<std::string, int64_t>> order(rhythmCount.begin(), rhythmCount.end());
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& z) {
        return a.second > z.second;
    });
    for(const auto& [k, v] : order) {
        SPDLOG_DEBUG("匹配格律：{} - {}", k, v);
    }
    order.clear();
    order.insert(order.begin(), unciCount.begin(), unciCount.end());
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& z) {
        return a.second > z.second;
    });
    for(const auto& [k, v] : order) {
        SPDLOG_DEBUG("未知词格律：{} - {}", k, v);
    }
    // order.clear();
    // order.insert(order.begin(), unshiCount.begin(), unshiCount.end());
    // std::sort(order.begin(), order.end(), [](const auto& a, const auto& z) {
    //     return a.second > z.second;
    // });
    // for(const auto& [k, v] : order) {
    //     SPDLOG_DEBUG("未知诗格律：{} - {}", k, v);
    // }
    SPDLOG_DEBUG("诗词总数：{} / {} / {}", fSize, count, total);
    SPDLOG_DEBUG("匹配词总数：{} / {}", ciCount, ciTotal);
    SPDLOG_DEBUG("匹配诗总数：{} / {}", shiCount, shiTotal);
}

LFR_TEST(
    // testPoetry();
    testMatchDataset();
);
