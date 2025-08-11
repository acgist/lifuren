#include "lifuren/Test.hpp"

#include "torch/torch.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"

[[maybe_unused]] static void testImage() {
    auto image { cv::imread(lifuren::file::join({ tmp_directory, "image.jpg" }).string()) };
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    cv::imshow("image", image);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image);
    cv::Mat target(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, CV_8UC3);
    lifuren::dataset::image::tensor_to_mat(target, tensor);
    cv::imshow("target", target);
    cv::waitKey();
    cv::destroyAllWindows();
}

[[maybe_unused]] static void testVideo() {
    cv::Mat src;
    cv::Mat dst;
    cv::VideoCapture video(lifuren::file::join({ tmp_directory, "wudaozi", "all", "BV1Wy54zMEyK.mp4" }).string());
    while(video.read(src) && video.read(dst)) {
        lifuren::dataset::image::resize(src, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        lifuren::dataset::image::resize(dst, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT * 2);
        cv::imshow("src", src);
        cv::imshow("dst", dst);
        cv::waitKey(10000);
    }
}

[[maybe_unused]] static void testLoadWudaoziDatasetLoader() {
    auto loader = lifuren::dataset::image::loadWudaoziDatasetLoader(
        LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT,
        20,
        lifuren::file::join({
            tmp_directory,
            "wudaozi",
            "train"
        }).string()
    );
    auto iterator = loader->begin();
    // SPDLOG_INFO("批次数量：{}", std::distance(iterator, loader->end()));
    std::cout << "视频特征数量\n" << iterator->data.sizes()   << std::endl;
    std::cout << "视频标签数量\n" << iterator->target.sizes() << std::endl;
    cv::Mat clone(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH * 2, CV_8UC3);
    cv::Mat frame(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH * 3, CV_8UC3);
    for(; iterator != loader->end(); ++iterator) {
        const int length = iterator->data.sizes()[0];
        for(int i = 0; i < length; ++i) {
            auto data   = iterator->data[i];
            auto target = iterator->target[i];
            if(data.count_nonzero().item<int>() == 0 || target.count_nonzero().item<int>() == 0) {
                cv::waitKey();
            }
            lifuren::dataset::image::tensor_to_mat(clone, data);
            // 注意：等于复制共享数据
            clone.copyTo(frame(cv::Rect(0, 0, LFR_IMAGE_WIDTH * 2, LFR_IMAGE_HEIGHT)));
            frame(cv::Rect(LFR_IMAGE_WIDTH * 2, 0, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT)) = clone(cv::Rect(0, 0, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT)) - clone(cv::Rect(LFR_IMAGE_WIDTH, 0, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT));
            cv::imshow("frame", frame);
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
    testLoadWudaoziDatasetLoader();
);
