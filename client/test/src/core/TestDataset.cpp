#include "lifuren/Test.hpp"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Dataset.hpp"

[[maybe_unused]] static void testImage() {
    auto image { cv::imread(lifuren::file::join({ lifuren::config::CONFIG.tmp, "image.jpg" }).string()) };
    cv::imshow("image", image);
    cv::waitKey();
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image);
    cv::Mat target(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, CV_8UC3);
    lifuren::dataset::image::tensor_to_mat(target, tensor);
    cv::imshow("target", target);
    cv::waitKey();
    cv::destroyAllWindows();
}

[[maybe_unused]] static void testVideo() {
    cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "w.mp4" }).string());
    // cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "h.mp4" }).string());
    cv::Mat frame;
    while(video.read(frame)) {
        lifuren::dataset::image::resize(frame, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
        cv::imshow("frame", frame);
        cv::waitKey(20);
    }
    cv::waitKey();
    cv::destroyAllWindows();
}

[[maybe_unused]] static void testLoadWudaoziDatasetLoader() {
    auto loader = lifuren::dataset::image::loadWudaoziDatasetLoader(
        LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT,
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
    cv::Mat image(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, CV_8UC3);
    for(; iterator != loader->end(); ++iterator) {
        const int length = iterator->data.sizes()[0];
        for(int i = 0; i < length; ++i) {
            auto tensor = iterator->data[i];
            lifuren::dataset::image::tensor_to_mat(image, tensor);
            cv::imshow("image", image);
            cv::waitKey(20);
        }
    }
}

LFR_TEST(
    // testImage();
    testVideo();
    // testLoadWudaoziDatasetLoader();
);
