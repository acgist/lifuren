#include "../header/Core.hpp"

void lifuren::testJson() {
    const nlohmann::json json{
        { "pi"   , 3.141 },
        { "happy", true  }
    };
    // const nlohmann::json json = R"(
    //     {
    //         "pi"   : 3.141,
    //         "happy": true
    //     }
    // )"_json;
    // const nlohmann::json json = nlohmann::json::parse(R"(
    // {
    //     "pi"   : 3.141,
    //     "happy": true
    // }
    // )");
    double pi = json["pi"];
    SPDLOG_DEBUG("pi = {}", pi);
    nlohmann::json array = nlohmann::json::array();
    const int ints[] = { 1, 2 };
    for(int index = 0; index < 2; index++) {
        array.push_back(ints[index]);
    }
    SPDLOG_DEBUG("int array = {}", array.dump());
    array.clear();
    const std::string strings[] = { "1", "2" };
    for(int index = 0; index < 2; index++) {
        array.push_back(strings[index]);
    }
    SPDLOG_DEBUG("string array = {}", array.dump());
    array.clear();
    nlohmann::json object = nlohmann::json::object();
    object["name"] = "碧螺萧萧";
    object.push_back({ "age", 4 });
    SPDLOG_DEBUG("object = {}", object.dump());
    object.clear();
    SPDLOG_DEBUG("");
}

void lifuren::testMark() {
    lifuren::Mark mark;
    mark.labels.push_back("acgist");
    mark.labels.push_back("lifuren");
    SPDLOG_DEBUG("mark = ", mark.toJSON());
    lifuren::MarkFile markFile;
    markFile.file = "lifuren.json";
    markFile.labels.push_back("acgist");
    markFile.labels.push_back("lifuren");
    SPDLOG_DEBUG("mark file = {}", markFile.toJSON());
    lifuren::MarkText* markTextPtr = new lifuren::MarkText();
    markTextPtr->name = "水调歌头";
    markTextPtr->text = "明月几时有 把酒问青天";
    markTextPtr->labels.push_back("acgist");
    markTextPtr->labels.push_back("lifuren");
    SPDLOG_DEBUG("mark text = {}", markTextPtr->toJSON());
    delete markTextPtr;
    SPDLOG_DEBUG("");
}

void lifuren::testLabel() {
    lifuren::Label label;
    label.name = "acgist";
    SPDLOG_DEBUG("label = {}", label.toJSON());
    lifuren::LabelFile labelFile;
    labelFile.name = "lifuren";
    labelFile.labels.push_back("acgist");
    labelFile.labels.push_back("lifuren");
    SPDLOG_DEBUG("label file = {}", labelFile.toJSON());
    lifuren::LabelText labelText;
    labelText.name = "text";
    SPDLOG_DEBUG("label text = {}", labelText.toJSON());
    lifuren::LabelText labelJson(R"(
        {"fontSize":0,"name":"text","segmentRule":[],"segmentSize":0}
    )");
    SPDLOG_DEBUG("label text name = {}", labelJson.name);
    SPDLOG_DEBUG("label text font size = {}", labelJson.fontSize);
    SPDLOG_DEBUG("label text segment rule = {}", labelJson.segmentRule.size());
    SPDLOG_DEBUG("");
}

void lifuren::testSetting() {
    lifuren::Setting setting;
    setting.modelPath = "路径";
    setting.activation = lifuren::Activation::RELU;
    SPDLOG_DEBUG("setting = {}", setting.toJSON());
    lifuren::Setting settingJson(R"(
        {"activation":0,"learningRate":0.01,"modelPath":"路径","regularization":0,"regularizationRate":0.01}
    )");
    SPDLOG_DEBUG("setting modelPath = {}", settingJson.modelPath);
    SPDLOG_DEBUG("setting activation = {}", settingJson.activation);
    SPDLOG_DEBUG("setting learning rate = {}", settingJson.learningRate);
    SPDLOG_DEBUG("setting regularization = {}", settingJson.regularization);
    SPDLOG_DEBUG("setting regularization rate = {}", settingJson.regularizationRate);
    lifuren::Settings settings;
    settings.load(R"(
        {
            "ImageGC": {
                "modelPath": "",
                "activation": 0,
                "learningRate": 0.01,
                "regularization": 1,
                "regularizationRate": 0.01
            },
            "ImageTS": {
                "modelPath": "",
                "activation": 0,
                "learningRate": 0.01
            }
        }
    )");
    std::map<std::string, lifuren::Setting>::iterator iterator = settings.settings.begin();
    std::map<std::string, lifuren::Setting>::iterator end      = settings.settings.end();
    for(; iterator != end; iterator++) {
        SPDLOG_DEBUG("key = {} value = {}", iterator->first, iterator->second.toJSON());
    }
    settings.settings["acgist"] = setting;
    SPDLOG_DEBUG("settings = {}", settings.toJSON());
    settings.saveFile("D:/tmp/settings.json");
    lifuren::Settings reload;
    reload.loadFile("D:/tmp/settings.json");
    assert(reload.toJSON() == settings.toJSON());
}
