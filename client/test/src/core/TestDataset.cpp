#include "lifuren/Test.hpp"

#include "torch/torch.h"

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
    // cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "w.mp4" }).string());
    // cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "h.mp4" }).string());
    cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "all", "BV1CtSKYQEKt.mp4" }).string());
    // cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "train", "BV1D1V7zQEV4.mp4" }).string());
    // cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "train", "BV1RYowY7EkK.mp4" }).string());
    double mean;
    cv::Mat old;
    cv::Mat diff;
    cv::Mat frame;
    cv::Mat source;
    int count = 0;
    while(video.read(frame)) {
        mean = cv::mean(frame)[0];
        if(mean < 10.0) {
            SPDLOG_INFO("黑屏：{}", mean);
            cv::imshow("black", frame);
            cv::waitKey();
            continue;
        }
        if(!old.empty()) {
            cv::absdiff(frame, old, diff);
            ++count;
            mean = cv::mean(diff)[0];
            SPDLOG_INFO("差异：{}", mean);
            if(mean == 0) {
                count = 0;
                continue;
            } else if(mean > LFR_VIDEO_DIFF) {
                SPDLOG_INFO("帧数：{}", count);
                count = 0;
                cv::waitKey();
            } else {
            }
            diff = frame - old;
            source = diff + old;
            lifuren::dataset::image::resize(diff, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
            lifuren::dataset::image::resize(source, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
            cv::imshow("diff", diff);
            cv::imshow("source", source);
        }
        old = frame;
        lifuren::dataset::image::resize(frame, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        cv::imshow("frame", frame);
        if(cv::waitKey(20) == 27) {
            break;
        }
    }
    cv::waitKey();
    cv::destroyAllWindows();
}

[[maybe_unused]] static void testLoadWudaoziDatasetLoader() {
    auto loader = lifuren::dataset::image::loadWudaoziDatasetLoader(
        LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT,
        20,
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
            if(tensor.count_nonzero().item<int>() == 0) {
                cv::waitKey();
            }
            lifuren::dataset::image::tensor_to_mat(image, tensor);
            cv::imshow("image", image);
            cv::waitKey(20);
        }
    }
    cv::waitKey();
    cv::destroyAllWindows();
}

LFR_TEST(
    // testImage();
    testVideo();
    // testLoadWudaoziDatasetLoader();
);
