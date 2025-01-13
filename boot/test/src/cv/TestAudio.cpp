#include "lifuren/Test.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/audio/Audio.hpp"

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

LFR_TEST(
    // testToPcm();
    // testToFile();
    testEmbedding();
);
