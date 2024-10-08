#include "lifuren/Test.hpp"

#include "nlohmann/json.hpp"

#include "lifuren/Poetrys.hpp"

[[maybe_unused]] static void testPoetrys() {
    nlohmann::json json = nlohmann::json::parse(R"(
    {
        "author": "欧阳修", 
        "paragraphs": [
            "庭院深深深几许，杨柳堆烟，帘幕无重数。",
            "玉勒雕鞍游冶处，楼高不见章台路。",
            "雨横风狂三月暮，门掩黄昏，无计留春住。",
            "泪眼问花花不语，乱红飞过秋千去。"
        ], 
        "rhythmic": "蝶恋花"
    }
    )");
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
    lifuren::poetrys::Poetry poetry = json;
    poetry.preproccess();
    SPDLOG_DEBUG("匹配格律：\n{}", poetry.matchRhythm());
    SPDLOG_DEBUG("原始段落：\n{}", poetry.segment);
    SPDLOG_DEBUG("朴素段落：\n{}", poetry.simpleSegment);
    if(poetry.matchRhythm()) {
        poetry.participle();
        SPDLOG_DEBUG("分词段落：\n{}", poetry.participleSegment);
    }
    lifuren::poetrys::Poetry diff = json;
    assert(diff == poetry);
}

LFR_TEST(
    testPoetrys();
);
