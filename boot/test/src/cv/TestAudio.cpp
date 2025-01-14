#include "lifuren/Test.hpp"

#include "torch/torch.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/audio/Audio.hpp"
#include "lifuren/audio/AudioDataset.hpp"

[[maybe_unused]] static void testToPcm() {
    // lifuren::audio::toPcm(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.aac"}).string());
    lifuren::audio::toPcm(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.mp3"}).string());
    // lifuren::audio::toPcm(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.flac"}).string());
}

[[maybe_unused]] static void testToFile() {
    lifuren::audio::toFile(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.pcm"}).string());
}

[[maybe_unused]] static void testEmbedding() {
    lifuren::dataset::allDatasetPreprocessing(
        lifuren::file::join({lifuren::config::CONFIG.tmp, "embedding"}).string(),
        lifuren::config::EMBEDDING_MODEL_FILE,
        &lifuren::audio::embedding
    );
}

[[maybe_unused]] static void testStftIstft() {
    std::ifstream input;
    std::ofstream output;
    input.open (lifuren::file::join({ lifuren::config::CONFIG.tmp, "noise.pcm"      }).string(), std::ios_base::binary);
    output.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "noise_copy.pcm" }).string(), std::ios_base::binary);
    std::vector<short> data;
    data.resize(DATASET_PCM_LENGTH);
    float norm_factor;
    while(input.read(reinterpret_cast<char*>(data.data()), DATASET_PCM_LENGTH * sizeof(short))) {
        auto tuple = std::move(lifuren::dataset::audio::pcm_mag_pha_stft(data, norm_factor));
        lifuren::logTensor("mag size", std::get<0>(tuple).sizes());
        lifuren::logTensor("pha size", std::get<1>(tuple).sizes());
        auto pcm   = std::move(lifuren::dataset::audio::pcm_mag_pha_istft(std::get<0>(tuple), std::get<1>(tuple), norm_factor));
        output.write(reinterpret_cast<char*>(pcm.data()), pcm.size() * sizeof(short));
        output.flush();
    }
    input.close();
    output.close();
}

LFR_TEST(
    // testToPcm();
    // testToFile();
    // testEmbedding();
    testStftIstft();
);
