#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

lifuren::dataset::DatasetLoader lifuren::dataset::image::loadChopinDatasetLoader(
    const int width,
    const int height,
    const size_t batch_size,
    const std::string& path
) {
    // TODO
    return {};
}

lifuren::dataset::DatasetLoader lifuren::dataset::image::loadMozartDatasetLoader(
    const int width,
    const int height,
    const size_t batch_size,
    const std::string& path
) {
    // TODO
    return {};
}

lifuren::dataset::DatasetLoader lifuren::dataset::image::loadWudaoziDatasetLoader(
    const int width,
    const int height,
    const size_t batch_size,
    const std::string& path
) {
    auto dataset = lifuren::dataset::Dataset(
        path,
        { ".jpg", ".png", ".jpeg" },
        [width, height] (
            const std::string         & file,
            std::vector<torch::Tensor>& labels,
            std::vector<torch::Tensor>& features,
            const torch::DeviceType   & device
        ) {
            auto image = cv::imread(file);
            lifuren::dataset::image::resize(image, width, height);
            auto tensor = lifuren::dataset::image::feature(image, width, height);
            // TODO
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

lifuren::dataset::DatasetLoader lifuren::dataset::image::loadClassifyDatasetLoader(
    const int width,
    const int height,
    const size_t batch_size,
    const std::string& path,
    const std::map<std::string, float>& classify
) {
    auto dataset = lifuren::dataset::Dataset(
        path,
        { ".jpg", ".png", ".jpeg" },
        classify,
        [width, height] (const std::string& file, const torch::DeviceType& device) -> torch::Tensor {
            auto image = cv::imread(file);
            lifuren::dataset::image::resize(image, width, height);
            auto tensor = lifuren::dataset::image::feature(image, width, height);
            return tensor.clone().to(device);
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

void lifuren::dataset::image::resize(cv::Mat& image, const int width, const int height) {
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

torch::Tensor lifuren::dataset::image::feature(
    const cv::Mat& image,
    const int width,
    const int height
) {
    if(image.empty()) {
        return {};
    }
    return torch::from_blob(image.data, { height, width, 3 }, torch::kByte).permute({2, 0, 1}).to(torch::kFloat32).div(255.0);
}

void lifuren::dataset::image::tensor_to_mat(cv::Mat& image, const torch::Tensor& tensor) {
    if(image.empty()) {
        return;
    }
    auto image_tensor = tensor.permute({1, 2, 0}).mul(255.0).to(torch::kByte).contiguous();
    std::memcpy(image.data, reinterpret_cast<char*>(image_tensor.data_ptr()), image.total() * image.elemSize());
}
