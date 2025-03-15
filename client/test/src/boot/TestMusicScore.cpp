#include "lifuren/Test.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/MusicScore.hpp"

[[maybe_unused]] static void testSaveLoad() {
    lifuren::music::Score score;
    score.name = "测试";
    auto path = lifuren::file::join({ lifuren::config::CONFIG.tmp, "music.xml" }).string();
    lifuren::music::save_xml(path, {});
    score = lifuren::music::load_xml(path);
    SPDLOG_DEBUG("乐谱名称：{}", score.name);
}

LFR_TEST(
    testSaveLoad();
);
