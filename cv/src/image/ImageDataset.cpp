#include "lifuren/image/ImageDataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

bool lifuren::image::read(const std::string& path, char* data, const size_t width, const size_t height) {
    auto image{ cv::imread(path) };
    return lifuren::image::read(image, data, width, height);
}

bool lifuren::image::read(cv::Mat& image, char* data, const size_t width, const size_t height) {
    if(image.total() <= 0LL) {
        return false;
    }
    cv::resize(image, image, cv::Size(width, height));
    std::memcpy(data, image.data, image.total() * image.elemSize());
    return true;
}

bool lifuren::image::write(const std::string& path, const char* data, const size_t width, const size_t height) {
    if(data == nullptr) {
        return false;
    }
    if(width == 0LL || height == 0LL) {
        return false;
    }
    lifuren::file::createParent(path);
    cv::Mat image(height, width, CV_8UC3);
    std::memcpy(image.data, data, width * height * 3);
    return cv::imwrite(path, image);
}

lifuren::dataset::FileDatasetLoader lifuren::image::loadFileDatasetLoader(
    const int width,
    const int height,
    const size_t batch_size,
    const std::string& path,
    const std::map<std::string, float>& classify
) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        { ".jpg", ".png", ".jpeg" },
        classify,
        [width, height] (const std::string& file, const torch::DeviceType& device) -> torch::Tensor {
            size_t length{ 0 };
            std::vector<char> feature;
            feature.resize(width * height * 3);
            lifuren::image::read(file, feature.data(), width, height);
            return lifuren::image::feature(feature.data(), width, height, device);
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}
