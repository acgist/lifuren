#include "lifuren/config/Label.hpp"

std::map<std::string, std::vector<lifuren::LabelFile>> lifuren::LABEL_AUDIO = lifuren::LabelFile::loadFile(lifuren::LABEL_AUDIO_PATH);
std::map<std::string, std::vector<lifuren::LabelFile>> lifuren::LABEL_IMAGE = lifuren::LabelFile::loadFile(lifuren::LABEL_IMAGE_PATH);
std::map<std::string, std::vector<lifuren::LabelFile>> lifuren::LABEL_VIDEO = lifuren::LabelFile::loadFile(lifuren::LABEL_VIDEO_PATH);
std::map<std::string, lifuren::LabelText> lifuren::LABEL_POETRY = lifuren::LabelText::loadFile(lifuren::LABEL_POETRY_PATH);

lifuren::Label::Label() {
}

lifuren::Label::Label(const std::string& name, const std::string& alias) : name(name), alias(alias) {
}

lifuren::Label::~Label() {
}
