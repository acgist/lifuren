#include "../header/Lifuren.hpp"

#include "../header/utils/Yamls.hpp"
#include "../header/config/Label.hpp"
#include "../header/config/Config.hpp"

void lifuren::loadConfig() {
    SPDLOG_DEBUG("加载配置");
    // 配置
    auto configs = lifuren::config::loadFile(lifuren::CONFIGS_PATH);
    lifuren::CONFIGS.clear();
    lifuren::CONFIGS.insert(configs.begin(), configs.end());
    // 音频
    auto audio = lifuren::LabelFile::loadFile(lifuren::LABEL_AUDIO_PATH);
    lifuren::LABEL_AUDIO.clear();
    lifuren::LABEL_AUDIO.insert(audio.begin(), audio.end());
    // 图片
    auto image = lifuren::LabelFile::loadFile(lifuren::LABEL_IMAGE_PATH);
    lifuren::LABEL_IMAGE.clear();
    lifuren::LABEL_IMAGE.insert(image.begin(), image.end());
    // 视频
    auto video = lifuren::LabelFile::loadFile(lifuren::LABEL_VIDEO_PATH);
    lifuren::LABEL_VIDEO.clear();
    lifuren::LABEL_VIDEO.insert(video.begin(), video.end());
    // 诗词
    auto poetry = lifuren::LabelText::loadFile(lifuren::LABEL_POETRY_PATH);
    lifuren::LABEL_POETRY.clear();
    lifuren::LABEL_POETRY.insert(poetry.begin(), poetry.end());
}