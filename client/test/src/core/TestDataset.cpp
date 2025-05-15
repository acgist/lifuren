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
    cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "all", "BV1kEVfzHExj.mp4" }).string());
    // cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "train", "BV1D1V7zQEV4.mp4" }).string());
    // cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "train", "BV1RYowY7EkK.mp4" }).string());
    cv::Mat old;
    cv::Mat diff;
    cv::Mat frame;
    int count = 0;
    while(video.read(frame)) {
        double min = 0;
        double max = 0;
        cv::minMaxLoc(frame, &min, &max);
        if(max == 0 && min == 0) {
            continue;
        }
        if(!old.empty()) {
            cv::absdiff(frame, old, diff);
            ++count;
            auto mean = cv::mean(diff)[0];
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
        }
        old = frame;
        lifuren::dataset::image::resize(frame, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
        cv::imshow("frame", frame);
        cv::waitKey(20);
    }
    cv::waitKey();
    cv::destroyAllWindows();
}

[[maybe_unused]] static void testAction() {
    cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "all", "BV1kEVfzHExj.mp4" }).string());
    cv::Mat frame;
    // cv::Mat image;
    std::vector<cv::Mat> images;
    while(video.read(frame)) {
        // cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);
        // cv::imshow("image", image);
        cv::split(frame, images);
        cv::imshow("image", images[2]);
        if(cv::waitKey(20) == 27) {
            break;
        }
    }
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
    // testAction();
    // testLoadWudaoziDatasetLoader();
);
