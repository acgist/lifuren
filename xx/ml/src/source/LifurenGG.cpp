#include "../header/LifurenGG.hpp"

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
    Mark mark;
    mark.file = "lifuren.json";
    LOG(INFO) << mark.toJSON();
}

void lifuren::testString() {
    std::string format = "li{}ren{}";
    std::string args[] = { "fu", "!!" };
    std::string flag   = "{}";
    lifuren::format(format, flag, args, 2);
    LOG(INFO) << format;
}
