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
    input.open (lifuren::file::join({ lifuren::config::CONFIG.tmp, "hhg.pcm"        }).string(), std::ios_base::binary);
    output.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "hhg_target.pcm" }).string(), std::ios_base::binary);
    // input.open (lifuren::file::join({ lifuren::config::CONFIG.tmp, "noise.pcm"        }).string(), std::ios_base::binary);
    // output.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "noise_target.pcm" }).string(), std::ios_base::binary);
    std::vector<short> data;
    data.resize(LFR_DATASET_PCM_LENGTH);
    while(input.read(reinterpret_cast<char*>(data.data()), LFR_DATASET_PCM_LENGTH * sizeof(short))) {
        // auto tensor = lifuren::audio::pcm_stft(data, 400, 40, 400);
        auto tensor = lifuren::audio::pcm_stft(data, 400, 80, 400);
        // auto tensor = lifuren::audio::pcm_stft(data, 400, 100, 400);
        // auto real   = torch::view_as_real(tensor);
        // auto compx  = torch::view_as_complex(real);
        // lifuren::logTensor("tensor size", real.sizes());
        // lifuren::logTensor("tensor size", compx.sizes());
        lifuren::logTensor("tensor size", tensor.sizes());
        // lifuren::logTensor("tensor size", real);
        // lifuren::logTensor("tensor size", compx);
        // lifuren::logTensor("tensor size", tensor);
        // auto pcm = lifuren::audio::pcm_istft(tensor, 400, 40, 400);
        auto pcm = lifuren::audio::pcm_istft(tensor, 400, 80, 400);
        // auto pcm = lifuren::audio::pcm_istft(tensor, 400, 100, 400);
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
    testEmbedding();
    // testLoadAudioFileDataset();
);
