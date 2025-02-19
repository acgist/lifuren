#include "lifuren/image/ImageDataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

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
            return lifuren::image::feature(cv::imread(file), width, height, device);
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

void lifuren::image::resize(cv::Mat& image, const int width, const int height) {
    const int cols = image.cols;
    const int rows = image.rows;
    const double ws = 1.0 * cols / width;
    const double hs = 1.0 * rows / height;
    const double scale = std::max(ws, hs);
    const int w = std::max(static_cast<int>(width  * scale), cols);
    const int h = std::max(static_cast<int>(height * scale), rows);
    cv::Mat result = cv::Mat::zeros(h, w, CV_8UC3);
    image.copyTo(result(cv::Rect(0, 0, cols, rows)));
    cv::resize(result, image, cv::Size(width, height));
}

torch::Tensor lifuren::image::feature(
    const cv::Mat& image,
    const int width,
    const int height,
    const torch::DeviceType& device
) {
    if(image.empty()) {
        return {};
    }
    cv::Mat target = image.clone();
    lifuren::image::resize(target, width, height);
    return torch::from_blob(target.data, { height, width, 3 }, torch::kByte).permute({2, 0, 1}).to(torch::kFloat32).div(255.0).to(device);
}

void lifuren::image::tensor_to_mat(cv::Mat& image, const torch::Tensor& tensor) {
    if(image.empty()) {
        return;
    }
    auto image_tensor = tensor.permute({2, 0, 1}).mul(255.0).to(torch::kByte);
    std::memcpy(image.data, reinterpret_cast<char*>(image_tensor.data_ptr()), image.total() * image.elemSize());
}
