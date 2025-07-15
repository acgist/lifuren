#include "lifuren/Test.hpp"

#include "torch/torch.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"

[[maybe_unused]] static void testImage() {
    auto image { cv::imread(lifuren::file::join({ lifuren::config::CONFIG.tmp, "image.jpg" }).string()) };
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    cv::imshow("image", image);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image);
    cv::Mat target(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, CV_8UC3);
    // tensor = tensor + torch::randn_like(tensor);
    // tensor = tensor - torch::randn_like(tensor);
    lifuren::dataset::image::tensor_to_mat(target, tensor);
    cv::imshow("target", target);
    cv::waitKey();
    cv::destroyAllWindows();
}

[[maybe_unused]] static void testVideo() {
    cv::Mat src;
    cv::Mat dst;
    cv::Mat pose(LFR_VIDEO_POSE_HEIGHT, LFR_VIDEO_POSE_WIDTH, CV_8UC1);
    cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "all", "BV1Wy54zMEyK.mp4" }).string());
    std::cout << lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "all", "BV1Wy54zMEyK.mp4" }).string() << std::endl;
    while(video.read(src) && video.read(dst)) {
        lifuren::dataset::image::resize(src, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        lifuren::dataset::image::resize(dst, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        auto tensor = lifuren::dataset::image::pose(pose, src, dst);
        lifuren::dataset::image::tensor_to_mat(pose, tensor);
        cv::Mat copy(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, CV_8UC1);
        cv::resize(pose, copy, cv::Size(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT), 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(copy, copy, cv::COLOR_GRAY2RGB);
        cv::imshow("src",  src);
        cv::imshow("dst",  dst);
        cv::imshow("copy", copy);
        cv::waitKey(10000);
    }
}

[[maybe_unused]] static void testNoise() {
    cv::Mat src;
    cv::Mat dst;
    cv::Mat diff_1(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 2, CV_8UC3);
    cv::Mat diff_2(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 2, CV_8UC3);
    cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "all", "BV1Wy54zMEyK.mp4" }).string());
    while(video.read(src) && video.read(dst)) {
        diff_1 = dst - src;
        lifuren::dataset::image::resize(src,    LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        lifuren::dataset::image::resize(dst,    LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        lifuren::dataset::image::resize(diff_1, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        auto src_tensor = lifuren::dataset::image::mat_to_tensor(src);
        auto dst_tensor = lifuren::dataset::image::mat_to_tensor(dst);
        auto noise = torch::rand_like(src_tensor);
        float ratio = 0.1;
        src_tensor = src_tensor * ratio + noise * (1 - ratio);
        dst_tensor = dst_tensor * ratio + noise * (1 - ratio);
        lifuren::dataset::image::tensor_to_mat(src, src_tensor);
        lifuren::dataset::image::tensor_to_mat(dst, dst_tensor);
        lifuren::dataset::image::tensor_to_mat(diff_2, src_tensor - dst_tensor);
        cv::imshow("src", src);
        cv::imshow("dst", dst);
        cv::imshow("diff_1", diff_1);
        cv::imshow("diff_2", diff_2);
        cv::waitKey();
    }
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
    std::cout << "视频特征数量\n" << iterator->data.sizes() << std::endl;
    std::cout << "视频标签数量\n" << iterator->target.sizes() << std::endl;
    cv::Mat pose (LFR_VIDEO_POSE_HEIGHT, LFR_VIDEO_POSE_WIDTH * 2, CV_8UC1);
    cv::Mat frame(LFR_IMAGE_HEIGHT,      LFR_IMAGE_WIDTH      * 2, CV_8UC3);
    cv::Mat show (LFR_IMAGE_HEIGHT * 2,  LFR_IMAGE_WIDTH      * 2, CV_8UC3);
    for(; iterator != loader->end(); ++iterator) {
        const int length = iterator->data.sizes()[0];
        for(int i = 0; i < length; ++i) {
            auto data   = iterator->data[i];
            auto target = iterator->target[i];
            if(data.count_nonzero().item<int>() == 0 || target.count_nonzero().item<int>() == 0) {
                cv::waitKey();
            }
            std::cout << target.slice(0, 0, 1).squeeze(0).select(0, 0).select(0, 0).to(torch::kLong) << std::endl;
            lifuren::dataset::image::tensor_to_mat(pose,  target.unsqueeze(1));
            lifuren::dataset::image::tensor_to_mat(frame, data);
            cv::Mat copy(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH * 2, CV_8UC1);
            cv::resize(pose, copy, cv::Size(LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT), 0, 0, cv::INTER_NEAREST);
            cv::cvtColor(copy, copy, cv::COLOR_GRAY2RGB);
            frame.copyTo(show(cv::Rect(0, 0 * LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT)));
            copy .copyTo(show(cv::Rect(0, 1 * LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT)));
            cv::imshow("show", show);
            if(cv::waitKey(10000) == 27) {
                break;
            }
        }
    }
    cv::waitKey();
    cv::destroyAllWindows();
}

LFR_TEST(
    // testImage();
    // testVideo();
    // testNoise();
    testLoadWudaoziDatasetLoader();
);
