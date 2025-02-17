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
                  int frame_index = 0;
            const int frame_count = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
            SPDLOG_INFO("加载视频文件：{} - {}", file, frame_count);
            cv::Mat frame;
            while(video.read(frame) && ++frame_index <= frame_count) {
                auto frame_feature = lifuren::image::feature(frame, width, height, device);
                if(frame_index == 1) {
                    features.push_back(frame_feature);
                } else if(frame_index == frame_count) {
                    labels.push_back(frame_feature);
                } else {
                    features.push_back(frame_feature);
                    labels.push_back(frame_feature);
                }
            }
            video.release();
            SPDLOG_INFO("视频加载完成：{} - {} / {}", file, frame_index, frame_count);
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}
