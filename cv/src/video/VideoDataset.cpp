#include "lifuren/video/VideoDataset.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/Config.hpp"
#include "lifuren/image/ImageDataset.hpp"

lifuren::dataset::FileDatasetLoader lifuren::video::loadFileDatasetLoader(
    const int width,
    const int height,
    const size_t batch_size,
    const std::string& path
) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        { ".mp4" },
        [width, height] (
            const std::string         & file,
            std::vector<torch::Tensor>& labels,
            std::vector<torch::Tensor>& features,
            const torch::DeviceType   & device
        ) -> void {
            cv::VideoCapture video(file);
            if(!video.isOpened()) {
                SPDLOG_WARN("视频文件打开失败：{}", file);
                video.release();
                return;
            }
            SPDLOG_INFO("加载视频文件：{}", file);
            int index = 0;
            cv::Mat frame;
            const int frame_count = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT)) - 1;
            while(video.read(frame)) {
                SPDLOG_INFO("{}", index);
                std::vector<char> feature;
                feature.resize(width * height * 3);
                lifuren::image::read(frame, feature.data(), width, height);
                auto frame_feature = lifuren::image::feature(feature.data(), width, height, device);
                if(index == 0) {
                    features.push_back(frame_feature);
                } else if(index == frame_count) {
                    labels.push_back(frame_feature);
                } else {
                    features.push_back(frame_feature);
                    labels.push_back(frame_feature);
                }
                ++index;
            }
            video.release();
            SPDLOG_INFO("视频加载完成：{} - {}", file, index);
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}
