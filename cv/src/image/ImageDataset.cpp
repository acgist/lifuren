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
            cv::Mat image = cv::imread(file);
            return lifuren::image::feature(image, width, height, device);
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

torch::Tensor lifuren::image::feature(
    const cv::Mat& image,
    const int width,
    const int height,
    const torch::DeviceType& type
) {
    if(image.empty()) {
        return {};
    }
    const int cols = image.cols;
    const int rows = image.rows;
    const double ws = 1.0 * cols / width;
    const double hs = 1.0 * rows / height;
    const double scale = std::max(ws, hs);
    const int w = width  * scale;
    const int h = height * scale;
    cv::Mat result = cv::Mat::zeros(h, w, CV_8UC3);
    image.copyTo(result(cv::Rect(0, 0, cols, rows)));
    cv::resize(result, result, cv::Size(width, height));
    return torch::from_blob(result.data, { height, width, 3 }, torch::kByte).permute({2, 0, 1}).to(torch::kFloat32).div(255.0).to(type);
}

void lifuren::image::tensor_to_mat(cv::Mat& image, const torch::Tensor& tensor) {
    if(image.empty()) {
        return;
    }
    auto image_tensor = tensor.permute({2, 0, 1}).mul(255.0).to(torch::kByte);
    std::memcpy(image.data, reinterpret_cast<char*>(image_tensor.data_ptr()), image.total() * image.elemSize());
}
