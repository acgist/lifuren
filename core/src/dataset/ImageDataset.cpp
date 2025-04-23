#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

void lifuren::dataset::image::resize(cv::Mat& image, const int width, const int height) {
    const int cols = image.cols;
    const int rows = image.rows;
    const double ws = 1.0 * cols / width;
    const double hs = 1.0 * rows / height;
    const double scale = std::min(ws, hs);
    const int w = std::max(static_cast<int>(width  * scale), cols);
    const int h = std::max(static_cast<int>(height * scale), rows);
    cv::Mat result = cv::Mat::zeros(h, w, CV_8UC3);
    image.copyTo(result(cv::Rect(0, 0, cols, rows)));
    cv::resize(result, image, cv::Size(width, height));
}

torch::Tensor lifuren::dataset::image::mat_to_tensor(const cv::Mat& image) {
    if(image.empty()) {
        return {};
    }
    return torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({2, 0, 1}).to(torch::kFloat32).div(255.0);
}

void lifuren::dataset::image::tensor_to_mat(cv::Mat& image, const torch::Tensor& tensor) {
    if(image.empty()) {
        return;
    }
    auto image_tensor = tensor.permute({1, 2, 0}).mul(255.0).to(torch::kByte).contiguous();
    std::memcpy(image.data, reinterpret_cast<char*>(image_tensor.data_ptr()), image.total() * image.elemSize());
}

lifuren::dataset::SeqDatasetLoader lifuren::dataset::image::loadWudaoziDatasetLoader(const int width, const int height, const size_t batch_size, const std::string& path) {
    auto dataset = lifuren::dataset::Dataset(
        path,
        ".xml",
        { ".png", ".jpg", ".jpeg" },
        [width, height] (
            const std::string         & l_file,
            const std::string         & f_file,
            std::vector<torch::Tensor>& labels,
            std::vector<torch::Tensor>& features,
            const torch::DeviceType   & device
        ) {
            // auto score = lifuren::music::load_xml(l_file);
            // auto image = cv::imread(f_file);
            // TODO: 分片：直线检测
            // TODO: 颜色二值化
            // auto l_tensor = lifuren::dataset::score::score_to_tensor(score);
            // lifuren::dataset::image::resize(image, width, height);
            // auto f_tensor = lifuren::dataset::image::mat_to_tensor(image);
            // labels.push_back(l_tensor.clone().to(device));
            // features.push_back(f_tensor.clone().to(device));
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<LFT_SEQ_SAMPLER>(std::move(dataset), batch_size);
}

lifuren::dataset::RndDatasetLoader lifuren::dataset::image::loadClassifyDatasetLoader(const int width, const int height, const size_t batch_size, const std::string& path, const std::map<std::string, float>& classify) {
    auto dataset = lifuren::dataset::Dataset(
        path,
        { ".png", ".jpg", ".jpeg" },
        classify,
        [width, height] (const std::string& file, const torch::DeviceType& device) -> torch::Tensor {
            auto image = cv::imread(file);
            if(image.empty()) {
                SPDLOG_WARN("加载数据失败：{}", file);
                return {};
            }
            lifuren::dataset::image::resize(image, width, height);
            auto tensor = lifuren::dataset::image::mat_to_tensor(image);
            return tensor.clone().to(device);
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<LFT_RND_SAMPLER>(std::move(dataset), batch_size);
}
