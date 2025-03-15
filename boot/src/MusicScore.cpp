#include "lifuren/MusicScore.hpp"

#include "tinyxml2.h"

#include "spdlog/spdlog.h"

lifuren::music::Score lifuren::music::load_xml(const std::string& path) {
    tinyxml2::XMLDocument doc;
    lifuren::music::Score score;
    if(doc.LoadFile(path.c_str()) != tinyxml2::XMLError::XML_SUCCESS) {
        SPDLOG_WARN("打开文件失败：{}", path);
        return score;
    }
    SPDLOG_DEBUG("打开文件：{}", path);
    auto root = doc.RootElement();
    return score;
}

bool lifuren::music::save_xml(const std::string& path, const lifuren::music::Score& score) {
    tinyxml2::XMLDocument doc;
    auto decl = doc.NewDeclaration();
    doc.InsertFirstChild(decl);
    auto root = doc.NewElement("score-partwise");
    root->SetAttribute("version", "4.0");
    doc.InsertEndChild(root);
    root->SetText("1234");
    if(doc.SaveFile(path.c_str()) == tinyxml2::XMLError::XML_SUCCESS) {
        SPDLOG_DEBUG("保存文件：{}", path);
        return true;
    } else {
        SPDLOG_WARN("保存文件失败：{}", path);
        return false;
    }
}
