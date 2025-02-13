#include "lifuren/Test.hpp"

#include "torch/torch.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/audio/AudioDataset.hpp"

[[maybe_unused]] static void testToPcm() {
    // lifuren::audio::toPcm(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.aac"}).string());
    lifuren::audio::toPcm(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.mp3"}).string());
    // lifuren::audio::toPcm(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.flac"}).string());
}

[[maybe_unused]] static void testToFile() {
    lifuren::audio::toFile(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.pcm"}).string());
}

[[maybe_unused]] static void testStftIstft() {
    std::ifstream input;
    std::ofstream output;
    input.open (lifuren::file::join({ lifuren::config::CONFIG.tmp, "noise.pcm"      }).string(), std::ios_base::binary);
    output.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "noise_copy.pcm" }).string(), std::ios_base::binary);
    std::vector<short> data;
    data.resize(DATASET_PCM_LENGTH);
    while(input.read(reinterpret_cast<char*>(data.data()), DATASET_PCM_LENGTH * sizeof(short))) {
        auto tuple = lifuren::audio::pcm_mag_pha_stft(data);
        // lifuren::logTensor("mag size", std::get<0>(tuple).sizes());
        // lifuren::logTensor("pha size", std::get<1>(tuple).sizes());
        auto pcm = lifuren::audio::pcm_mag_pha_istft(std::get<0>(tuple), std::get<1>(tuple));
        output.write(reinterpret_cast<char*>(pcm.data()), pcm.size() * sizeof(short));
    }
    input.close();
    output.close();
}

[[maybe_unused]] static void testEmbedding() {
    lifuren::dataset::allDatasetPreprocessing(
        lifuren::file::join({lifuren::config::CONFIG.tmp, "baicai"}).string(),
        lifuren::config::EMBEDDING_MODEL_FILE,
        &lifuren::audio::embedding
    );
}

[[maybe_unused]] static void testLoadAudioFileDataset() {
    // 注意：如果需要还原不要使用RandomSampler而要使用SequentialSampler
    auto loader = lifuren::audio::loadFileDatasetLoader(200, lifuren::file::join({
        lifuren::config::CONFIG.tmp,
        "baicai",
        "train",
        lifuren::config::LIFUREN_HIDDEN_FILE,
        lifuren::config::EMBEDDING_MODEL_FILE
    }).string());
    lifuren::logTensor("音频特征", loader->begin()->data.sizes());
    lifuren::logTensor("音频标签", loader->begin()->target.sizes());
    SPDLOG_INFO("批次数量：{}", std::distance(loader->begin(), loader->end()));
}

LFR_TEST(
    // testToPcm();
    // testToFile();
    // testStftIstft();
    // testEmbedding();
    testLoadAudioFileDataset();
);
