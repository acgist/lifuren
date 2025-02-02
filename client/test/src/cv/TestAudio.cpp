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
    lifuren::audio::toFile(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "d.pcm"}).string());
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
        output.flush();
    }
    input.close();
    output.close();
}

[[maybe_unused]] static void testEmbedding() {
    lifuren::dataset::allDatasetPreprocessing(
        lifuren::file::join({lifuren::config::CONFIG.tmp, "audio"}).string(),
        lifuren::config::EMBEDDING_MODEL_FILE,
        &lifuren::audio::embedding
    );
}

[[maybe_unused]] static void testLoadAudioFileDataset() {
    auto loader = lifuren::audio::loadFileDatasetLoader(200, lifuren::file::join({
        lifuren::config::CONFIG.tmp,
        "audio",
        "train",
        lifuren::config::LIFUREN_HIDDEN_FILE,
        lifuren::config::EMBEDDING_MODEL_FILE
    }).string());
    lifuren::logTensor("音频特征", loader->begin()->data.sizes());
    lifuren::logTensor("音频标签", loader->begin()->target.sizes());
    // 注意：不要使用RandomSampler而要使用SequentialSampler
    // std::ofstream output;
    // output.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "audio.dataset.pcm" }).string(), std::ios_base::binary);
    // for(auto iter = loader->begin(); iter != loader->end(); ++iter) {
    //     auto data = iter->data;
    //     for(int i = 0; i < data.sizes()[0]; ++i) {
    //         auto pcm = lifuren::audio::pcm_mag_pha_istft(data[i].slice(0, 0, 1), data[i].slice(0, 1, 2));
    //         output.write(reinterpret_cast<char*>(pcm.data()), pcm.size() * sizeof(short));
    //     }
    // }
    // output.close();
}

LFR_TEST(
    // testToPcm();
    // testToFile();
    // testStftIstft();
    // testEmbedding();
    testLoadAudioFileDataset();
);
