#include "lifuren/Test.hpp"

#include "torch/torch.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"

[[maybe_unused]] static void testImage() {
    auto image { cv::imread(lifuren::file::join({ lifuren::config::CONFIG.tmp, "image.jpg" }).string()) };
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    cv::imshow("image", image);
    cv::waitKey();
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
    // cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "w.mp4" }).string());
    // cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "h.mp4" }).string());
    cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "all", "BV1Wy54zMEyK.mp4" }).string());
    // cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "train", "BV1D1V7zQEV4.mp4" }).string());
    // cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "train", "BV1RYowY7EkK.mp4" }).string());
    double mean;
    cv::Mat diff;
    cv::Mat prev;
    cv::Mat next;
    cv::Mat orig;
    int count = 0;
    cv::Mat show(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 2 * 3, CV_8UC3);
    while(video.read(next)) {
        mean = cv::mean(next)[0];
        if(mean < LFR_VIDEO_BLACK_MEAN) {
            SPDLOG_INFO("黑屏：{}", mean);
            continue;
        }
        if(!prev.empty()) {
            cv::absdiff(next, prev, diff);
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
            diff = next - prev;
            orig = diff + prev;
            cv::Mat pose(LFR_VIDEO_POSE_HEIGHT, LFR_VIDEO_POSE_WIDTH, CV_8UC1);
            lifuren::dataset::image::pose(pose, prev, next);
            cv::resize(pose, pose, cv::Size(LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2), 0, 0, cv::INTER_NEAREST);
            cv::cvtColor(pose, pose, cv::COLOR_GRAY2RGB);
            lifuren::dataset::image::resize(diff, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
            lifuren::dataset::image::resize(orig, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
            pose.copyTo(show(cv::Rect(0 * LFR_IMAGE_WIDTH * 2, 0, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2)));
            diff.copyTo(show(cv::Rect(1 * LFR_IMAGE_WIDTH * 2, 0, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2)));
            orig.copyTo(show(cv::Rect(2 * LFR_IMAGE_WIDTH * 2, 0, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2)));
            cv::imshow("frame", show);
            if(cv::waitKey(20) == 27) {
                break;
            }
        }
        prev = next;
        lifuren::dataset::image::resize(next, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
    }
    cv::waitKey();
    cv::destroyAllWindows();
}

[[maybe_unused]] static void testNoise() {
    cv::VideoCapture video(lifuren::file::join({ lifuren::config::CONFIG.tmp, "wudaozi", "all", "BV1Wy54zMEyK.mp4" }).string());
    cv::Mat src;
    cv::Mat dst;
    cv::Mat diff_1(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 2, CV_8UC3);
    cv::Mat diff_2(LFR_IMAGE_HEIGHT * 2, LFR_IMAGE_WIDTH * 2, CV_8UC3);
    while(video.read(src) && video.read(dst)) {
        diff_1 = dst - src;
        lifuren::dataset::image::resize(src, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        lifuren::dataset::image::resize(dst, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
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

[[maybe_unused]] static void testReshape() {
    int w       = 180;
    int h       = 320;
    int batch   = 1;
    int channel = 3;
    int w_scale = 6;
    int h_scale = 8;
    auto image = cv::imread(lifuren::file::join({ lifuren::config::CONFIG.tmp, "bike.jpg" }).string());
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    auto input = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0).unsqueeze(0).unsqueeze(0);
    input = input
        .reshape({ batch, channel, h_scale,           h / h_scale, w           }).permute({ 0, 1, 2, 4, 3 })
        .reshape({ batch, channel, h_scale * w_scale, w / w_scale, h / h_scale }).permute({ 0, 1, 2, 4, 3 })
        .permute({ 0, 2, 1, 3, 4 });
    image = cv::Mat(40, 30, CV_8UC3);
    for(int i = 0; i < 48; ++i) {
        auto tensor = input.slice(1, i, i + 1).squeeze();
        lifuren::dataset::image::tensor_to_mat(image, tensor);
        cv::imshow("image", image);
        cv::waitKey();
    }
    std::cout << input.sizes() << std::endl;
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
    std::cout << "视频特征数量：" << iterator->data.sizes() << std::endl;
    std::cout << "视频标签数量：" << iterator->target.sizes() << std::endl;
    cv::Mat pose (LFR_VIDEO_POSE_HEIGHT, LFR_VIDEO_POSE_WIDTH * 2, CV_8UC1);
    cv::Mat frame(LFR_IMAGE_HEIGHT,      LFR_IMAGE_WIDTH      * 2, CV_8UC3);
    for(; iterator != loader->end(); ++iterator) {
        const int length = iterator->data.sizes()[0];
        for(int i = 0; i < length; ++i) {
            auto data   = iterator->data[i];
            auto target = iterator->target[i];
            if(data.count_nonzero().item<int>() == 0 || target.count_nonzero().item<int>() == 0) {
                cv::waitKey();
            }
            lifuren::dataset::image::tensor_to_mat(pose,  target);
            lifuren::dataset::image::tensor_to_mat(frame, data);
            cv::imshow("pose",  pose);
            cv::imshow("frame", frame);
            cv::waitKey(1000);
        }
    }
    cv::waitKey();
    cv::destroyAllWindows();
}

LFR_TEST(
    // testImage();
    // testVideo();
    // testNoise();
    // testReshape();
    testLoadWudaoziDatasetLoader();
);
