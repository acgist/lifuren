#include "lifuren/Test.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/video/VideoDataset.hpp"

[[maybe_unused]] static void testLoadVideoFileDataset() {
    auto loader = lifuren::video::loadFileDatasetLoader(640, 640, 200, lifuren::file::join({lifuren::config::CONFIG.tmp, "video", "train"}).string());
    lifuren::logTensor("视频特征", loader->begin()->data.sizes());
    lifuren::logTensor("视频标签", loader->begin()->target.sizes());
}

LFR_TEST(
    testLoadVideoFileDataset();
);
