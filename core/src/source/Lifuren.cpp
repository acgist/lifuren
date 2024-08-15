#include "lifuren/Lifuren.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Yamls.hpp"
#include "lifuren/config/Label.hpp"
#include "lifuren/config/Config.hpp"

void lifuren::loadConfig() noexcept {
    SPDLOG_DEBUG("加载全局所有配置");
    // 配置
    auto config = lifuren::config::loadFile(lifuren::config::CONFIG_PATH);
    lifuren::config::CONFIG = config;
    // 音频标签
    auto audio = lifuren::LabelFile::loadFile(lifuren::LABEL_AUDIO_PATH);
    lifuren::LABEL_AUDIO.clear();
    // std::swap(lifuren::LABEL_AUDIO, audio);
    lifuren::LABEL_AUDIO.insert(audio.begin(), audio.end());
    // 图片标签
    auto image = lifuren::LabelFile::loadFile(lifuren::LABEL_IMAGE_PATH);
    lifuren::LABEL_IMAGE.clear();
    // std::swap(lifuren::LABEL_IMAGE, image);
    lifuren::LABEL_IMAGE.insert(image.begin(), image.end());
    // 视频标签
    auto video = lifuren::LabelFile::loadFile(lifuren::LABEL_VIDEO_PATH);
    lifuren::LABEL_VIDEO.clear();
    // std::swap(lifuren::LABEL_VIDEO, video);
    lifuren::LABEL_VIDEO.insert(video.begin(), video.end());
    // 诗词标签
    auto poetry = lifuren::LabelText::loadFile(lifuren::LABEL_POETRY_PATH);
    lifuren::LABEL_POETRY.clear();
    // std::swap(lifuren::LABEL_POETRY, poetry);
    lifuren::LABEL_POETRY.insert(poetry.begin(), poetry.end());
}