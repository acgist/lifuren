#include "lifuren/Test.hpp"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Dataset.hpp"

[[maybe_unused]] static void testImage() {
    auto image { cv::imread(lifuren::file::join({ lifuren::config::CONFIG.tmp, "image.jpg" }).string()) };
    cv::imshow("image", image);
    cv::waitKey();
    lifuren::dataset::image::resize(image, 640, 480);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image);
    cv::Mat target(480, 640, CV_8UC3);
    lifuren::dataset::image::tensor_to_mat(target, tensor);
    cv::imshow("target", target);
    cv::waitKey();
    cv::destroyAllWindows();
}

[[maybe_unused]] static void testLoadWudaoziDatasetLoader() {
    auto loader = lifuren::dataset::image::loadWudaoziDatasetLoader(
        640, 480,
        200,
        lifuren::file::join({
            lifuren::config::CONFIG.tmp,
            "wudaozi",
            "train"
        }).string()
    );
    auto iterator = loader->begin();
    // SPDLOG_INFO("批次数量：{}", std::distance(iterator, loader->end()));
    lifuren::logTensor("视频特征数量", iterator->data.sizes());
    lifuren::logTensor("视频标签数量", iterator->target.sizes());
    cv::Mat image(480, 640, CV_8UC3);
    const int length = iterator->data.sizes()[0];
    for(; iterator != loader->end(); ++iterator) {
        for(int i = 0; i < length; ++i) {
            auto tensor = iterator->data[i];
            lifuren::dataset::image::tensor_to_mat(image, tensor);
            cv::imshow("image", image);
            cv::waitKey(20);
        }
    }
}

LFR_TEST(
    // testPcm();
    // testStft();
    // testImage();
    testLoadWudaoziDatasetLoader();
);
