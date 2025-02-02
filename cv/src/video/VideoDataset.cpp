#include "lifuren/video/VideoDataset.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/Config.hpp"
#include "lifuren/image/ImageDataset.hpp"

lifuren::dataset::FileDatasetLoader lifuren::video::loadFileDatasetLoader(
    const int& width,
    const int& height,
    const size_t& batch_size,
    const std::string& path
) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        { ".mp4" },
        [width, height] (const std::string& file, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) -> void {
            cv::VideoCapture video(file);
            if(!video.isOpened()) {
                SPDLOG_WARN("视频文件打开失败：{}", file);
                video.release();
                return;
            }
            const int& frame_length = lifuren::config::CONFIG.video.length;
            int index = 0;
            cv::Mat frame;
            std::vector<torch::Tensor> tensors;
            tensors.reserve(frame_length);
            while(video.read(frame)) {
                std::vector<char> feature;
                feature.resize(width * height * 3);
                lifuren::image::read(frame, feature.data(), width, height);
                if(++index <= frame_length) {
                    tensors.push_back(std::move(lifuren::image::feature(feature.data(), width, height, device)));
                } else {
                    auto next = lifuren::image::feature(feature.data(), width, height, device);
                    features.push_back(torch::stack(tensors));
                    labels.push_back(next);
                    tensors.erase(tensors.begin());
                    tensors.push_back(next);
                }
            }
            video.release();
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}
