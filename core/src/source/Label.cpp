#include "../header/Label.hpp"

lifuren::LabelFile lifuren::LABEL_AUDIO;
lifuren::LabelFile lifuren::LABEL_IMAGE;
lifuren::LabelFile lifuren::LABEL_VIDEO;
lifuren::LabelText lifuren::LABEL_POETRY;

lifuren::Label::Label() {
}

lifuren::Label::~Label() {
}

lifuren::Label::Label(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::Label::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

void lifuren::Label::loadFile(const std::string& path) {
    std::string label = lifuren::files::loadFile(path);
    if(label.empty()) {
        return;
    }
    SPDLOG_DEBUG("加载标签文件：{}", path);
    SPDLOG_DEBUG("加载标签内容：{}", label);
    *this = nlohmann::json::parse(label);
}
