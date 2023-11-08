#include "../header/Boot.hpp"

void lifuren::testJson() {
    // const nlohmann::json json = nlohmann::json::parse(R"(
    // {
    //     "pi"   : 3.141,
    //     "happy": true
    // }
    // )");
    // const nlohmann::json json = R"(
    //     {
    //         "pi"   : 3.141,
    //         "happy": true
    //     }
    // )"_json;
    const nlohmann::json json{
        { "pi"   , 3.141 },
        { "happy", true  }
    };
    LOG(INFO) << "pi = " << json["pi"];
    nlohmann::json array = nlohmann::json::array();
    const int ints[] = { 1, 2 };
    for(int index = 0; index < 2; index++) {
        array.push_back(ints[index]);
    }
    LOG(INFO) << "int array = " << array.dump();
    array.clear();
    const std::string strings[] = { "1", "2" };
    for(int index = 0; index < 2; index++) {
        array.push_back(strings[index]);
    }
    LOG(INFO) << "string array = " << array.dump();
    array.clear();
    nlohmann::json object = nlohmann::json::object();
    object["name"] = "碧螺萧萧";
    object.push_back({ "age", 4 });
    LOG(INFO) << "object = " << object.dump();
}

void lifuren::testMark() {
    lifuren::Mark mark;
    mark.file = "lifuren.json";
    mark.labels.push_back("acgist");
    mark.labels.push_back("lifuren");
    LOG(INFO) << "mark = " << mark.toJSON();
    lifuren::MarkFile markFile;
    markFile.file = "lifuren.json";
    markFile.labels.push_back("acgist");
    markFile.labels.push_back("lifuren");
    LOG(INFO) << "mark file = " << markFile.toJSON();
    lifuren::MarkText* markTextPtr = new lifuren::MarkText();
    markTextPtr->name = "水调歌头";
    markTextPtr->text = "明月几时有 把酒问青天";
    markTextPtr->file = "lifuren.json";
    markTextPtr->labels.push_back("acgist");
    markTextPtr->labels.push_back("lifuren");
    LOG(INFO) << "mark text = " << markTextPtr->toJSON();
    delete markTextPtr;
}

void lifuren::testLabel() {
    lifuren::Label label;
    label.name = "acgist";
    LOG(INFO) << "label = " << label.toJSON();
    lifuren::LabelConfig labelConfig;
    labelConfig.name = "lifuren";
    labelConfig.labels.push_back("acgist");
    labelConfig.labels.push_back("lifuren");
    LOG(INFO) << "label config = " << labelConfig.toJSON();
    lifuren::LabelSegment labelSegment;
    labelSegment.name = "segment";
    LOG(INFO) << "label segment = " << labelSegment.toJSON();
    lifuren::LabelSegment labelJson(R"(
        {"fontSize":0,"name":"segment","segmentRule":[],"segmentSize":0}
    )");
    LOG(INFO) << "label segment name = " << labelJson.name;
    LOG(INFO) << "label segment font size = " << labelJson.fontSize;
    LOG(INFO) << "label segment segment rule = " << labelJson.segmentRule.size();
}

void lifuren::testSetting() {
    lifuren::Setting setting;
    setting.path = "路径";
    setting.activation = lifuren::Activation::RELU;
    LOG(INFO) << "setting = " << setting.toJSON();
    lifuren::Setting settingJson(R"(
        {"activation":0,"learningRate":0.01,"path":"路径","regularization":0,"regularizationRate":0.01}
    )");
    LOG(INFO) << "setting path = " << settingJson.path;
    LOG(INFO) << "setting activation = " << settingJson.activation;
    LOG(INFO) << "setting learning rate = " << settingJson.learningRate;
    LOG(INFO) << "setting regularization = " << settingJson.regularization;
    LOG(INFO) << "setting regularization rate = " << settingJson.regularizationRate;
    lifuren::Settings settings;
    settings.load(R"(
        {
            "ImageGC": {
                "path": "",
                "activation": 0,
                "learningRate": 0.01,
                "regularization": 1,
                "regularizationRate": 0.01
            },
            "ImageTS": {
                "path": "",
                "activation": 0,
                "learningRate": 0.01
            }
        }
    )");
    std::map<std::string, lifuren::Setting>::iterator iterator = settings.settings.begin();
    std::map<std::string, lifuren::Setting>::iterator end = settings.settings.end();
    for(; iterator != end; iterator++) {
        LOG(INFO) << "key = " << iterator->first << " value = " << iterator->second.toJSON();
    }
    settings.settings["acgist"] = setting;
    LOG(INFO) << "settings = " << settings.toJSON();
}
