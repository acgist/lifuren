#include "lifuren/Test.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/video/VideoDataset.hpp"

[[maybe_unused]] static void testLoadVideoFileDataset() {
    auto loader = lifuren::video::loadFileDatasetLoader(640, 480, 8, lifuren::file::join({lifuren::config::CONFIG.tmp, "video", "train"}).string());
    lifuren::logTensor("视频特征数量", loader->begin()->data.sizes());
    lifuren::logTensor("视频标签数量", loader->begin()->target.sizes());
    SPDLOG_INFO("批次数量：{}", std::distance(loader->begin(), loader->end()));
}

LFR_TEST(
    testLoadVideoFileDataset();
);
