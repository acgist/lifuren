#include "../header/Boot.hpp"

void lifuren::testJson() {
    nlohmann::json json = nlohmann::json::parse(R"(
    {
        "pi"   : 3.141,
        "happy": true
    }
    )");
    LOG(INFO) << json["pi"];
    std::string strings[] = { "1", "2" };
    nlohmann::json array = nlohmann::json::array();
    for(int index = 0; index < 2; index++) {
        array.push_back(strings[index]);
    }
    LOG(INFO) << array.dump();
    int ints[] = { 1, 2, 3 };
    array.clear();
    for(int index = 0; index < 2; index++) {
        array.push_back(ints[index]);
    }
    LOG(INFO) << array.dump();
}

void lifuren::testMark() {
    lifuren::Mark mark;
    mark.file = "lifuren.json";
    mark.labels.push_back("acgist");
    mark.labels.push_back("lifuren");
    LOG(INFO) << mark.toJSON();
    lifuren::MarkFile markFile;
    markFile.file = "lifuren.json";
    markFile.labels.push_back("acgist");
    markFile.labels.push_back("lifuren");
    LOG(INFO) << markFile.toJSON();
    lifuren::MarkText markText;
    markText.name = "水调歌头";
    markText.text = "明月几时有 把酒问青天";
    markText.file = "lifuren.json";
    markText.labels.push_back("acgist");
    markText.labels.push_back("lifuren");
    LOG(INFO) << markText.toJSON();
}

void lifuren::testLabel() {
    lifuren::Label label;
    label.name = "acgist";
    LOG(INFO) << label.toJSON();
    lifuren::LabelConfig labelConfig;
    labelConfig.name = "lifuren";
    labelConfig.labels.push_back("acgist");
    labelConfig.labels.push_back("lifuren");
    LOG(INFO) << labelConfig.toJSON();
    lifuren::LabelSegment labelSegment;
    labelSegment.name = "segment";
    LOG(INFO) << labelSegment.toJSON();
    lifuren::LabelSegment labelJson(R"(
        {"fontSize":0,"lifuren::Label::name":"segment","segmentRule":[],"segmentSize":0}
    )");
    LOG(INFO) << labelJson.name;
    LOG(INFO) << labelJson.fontSize;
    LOG(INFO) << labelJson.segmentRule.size();
}
