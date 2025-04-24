#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

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
        { ".mp4" },
        [width, height] (
            const std::string         & file,
            std::vector<torch::Tensor>& labels,
            std::vector<torch::Tensor>& features,
            const torch::DeviceType   & device
        ) {
            cv::VideoCapture video;
            video.open(file);
            if(!video.isOpened()) {
                SPDLOG_WARN("加载视频文件失败：{}", file);
                return;
            }
            size_t count = 0;
            cv::Mat frame;
            while(video.read(frame)) {
                ++count;
                lifuren::dataset::image::resize(frame, width, height);
                auto tensor = lifuren::dataset::image::mat_to_tensor(frame);
                labels  .push_back(tensor.clone().to(device));
                features.push_back(tensor.clone().to(device));
            }
            SPDLOG_DEBUG("加载视频文件帧数：{} - {}", file, count);
            video.release();
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<LFT_SEQ_SAMPLER>(std::move(dataset), batch_size);
}
