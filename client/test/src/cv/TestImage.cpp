#include "lifuren/Test.hpp"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/image/ImageDataset.hpp"

[[maybe_unused]] static void testFeature() {
    auto image { cv::imread(lifuren::file::join({ lifuren::config::CONFIG.tmp, "xxc.png" }).string()) };
    cv::imshow("image", image);
    cv::waitKey();
    auto tensor = lifuren::image::feature(image, 640, 480, torch::DeviceType::CPU);
    cv::Mat target(480, 640, CV_8UC3);
    lifuren::image::tensor_to_mat(target, tensor);
    cv::imshow("target", target);
    cv::waitKey();
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
    SPDLOG_INFO("数据大小：{}", std::distance(loader->begin(), loader->end()));
}

LFR_TEST(
    // testFeature();
    testLoadFileDatasetLoader();
);
