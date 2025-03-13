#include "lifuren/Test.hpp"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/image/ImageDataset.hpp"

[[maybe_unused]] static void testToPcm() {
    lifuren::audio::toPcm(lifuren::file::join({lifuren::config::CONFIG.tmp, "tts.mp3"}).string());
    // lifuren::audio::toPcm(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.aac"}).string());
    // lifuren::audio::toPcm(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.mp3"}).string());
    // lifuren::audio::toPcm(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.flac"}).string());
}

[[maybe_unused]] static void testToFile() {
    lifuren::audio::toFile(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "audio.pcm"}).string());
}

[[maybe_unused]] static void testStftIstft() {
    std::ifstream input;
    std::ofstream output;
    // input.open (lifuren::file::join({ lifuren::config::CONFIG.tmp, "noise.pcm"        }).string(), std::ios_base::binary);
    // output.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "noise_target.pcm" }).string(), std::ios_base::binary);
    input.open (lifuren::file::join({ lifuren::config::CONFIG.tmp, "baicai.pcm"        }).string(), std::ios_base::binary);
    output.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "baicai_target.pcm" }).string(), std::ios_base::binary);
    std::vector<short> data;
    data.resize(LFR_DATASET_PCM_LENGTH);
    // auto norm = torch::nn::BatchNorm2d(LFR_DATASET_PCM_DIM_1);
    while(input.read(reinterpret_cast<char*>(data.data()), LFR_DATASET_PCM_LENGTH * sizeof(short))) {
        // auto tensor = lifuren::audio::pcm_stft(data, 400, 40, 400);
        auto tensor = lifuren::audio::pcm_stft(data, 400, 80, 400);
        // auto tensor  = lifuren::audio::pcm_stft(data, 400, 100, 400);
        // auto real    = torch::view_as_real(tensor);
        // auto complex = torch::view_as_complex(real);
        // lifuren::logTensor("tensor size", tensor.sizes());
        // lifuren::logTensor("tensor size", real.sizes());
        // lifuren::logTensor("tensor size", complex.sizes());
        // lifuren::logTensor("tensor", tensor);
        // lifuren::logTensor("tensor", real);
        // lifuren::logTensor("tensor", complex);
        // lifuren::logTensor("tensor", norm->forward(tensor));
        // auto pcm = lifuren::audio::pcm_istft(tensor, 400, 40, 400);
        // cv::Mat image(201, 56, CV_8UC1, reinterpret_cast<char*>(tensor.data_ptr()));
        // image = image.t();
        // cv::imshow("image", image);
        // cv::waitKey();
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
    // SPDLOG_INFO("批次数量：{}", std::distance(loader->begin(), loader->end()));
}

[[maybe_unused]] static void testFeature() {
    auto image { cv::imread(lifuren::file::join({ lifuren::config::CONFIG.tmp, "xxc.png" }).string()) };
    cv::imshow("image", image);
    cv::waitKey();
    lifuren::image::resize(image, 640, 480);
    auto tensor = lifuren::image::feature(image, 640, 480);
    cv::Mat target(480, 640, CV_8UC3);
    lifuren::image::tensor_to_mat(target, tensor);
    cv::imshow("target", target);
    cv::waitKey();
    cv::destroyAllWindows();
}

[[maybe_unused]] static void testLoadFileDatasetLoader() {
    auto loader = lifuren::image::loadFileDatasetLoader(
        200,
        200,
        5,
        lifuren::file::join({lifuren::config::CONFIG.tmp, "gender", "train"}).string(),
        {
            { "man",   1.0F },
            { "woman", 0.0F }
        }
    );
    lifuren::logTensor("图片特征", loader->begin()->data.sizes());
    lifuren::logTensor("图片标签", loader->begin()->target.sizes());
    SPDLOG_INFO("批次数量：{}", std::distance(loader->begin(), loader->end()));
}

[[maybe_unused]] static void testLoadVideoFileDataset() {
    auto loader = lifuren::video::loadFileDatasetLoader(8, lifuren::file::join({lifuren::config::CONFIG.tmp, "video", "train"}).string());
    lifuren::logTensor("视频特征数量", loader->begin()->data.sizes());
    lifuren::logTensor("视频标签数量", loader->begin()->target.sizes());
    // SPDLOG_INFO("批次数量：{}", std::distance(loader->begin(), loader->end()));
    // 注意：如果需要还原不要使用RandomSampler而要使用SequentialSampler
    cv::Mat mat(LFR_VIDEO_HEIGHT, LFR_VIDEO_WIDTH, CV_8UC3);
    auto iterator = loader->begin();
    const int length = iterator->data.sizes()[0];
    for(; iterator != loader->end(); ++iterator) {
        for(int i = 0; i < length; ++i) {
            auto tensor = iterator->data[i];
            lifuren::image::tensor_to_mat(mat, tensor);
            cv::imshow("mat", mat);
            cv::waitKey(30);
        }
    }
}

LFR_TEST(
    testFeature();
    // testLoadFileDatasetLoader();
);
