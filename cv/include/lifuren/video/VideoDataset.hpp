/**
 * 视频数据集
 */
#ifndef LFR_HEADER_CV_VIDEO_DATASET_HPP
#define LFR_HEADER_CV_VIDEO_DATASET_HPP

#include "lifuren/Dataset.hpp"

namespace lifuren::video {

extern void feature(const int& width, const int& height, const std::string& file,                              std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device);
extern void feature(const int& width, const int& height, const std::string& source, const std::string& target, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device);

inline lifuren::dataset::FileDatasetLoader loadFileDatasetLoader(
    const int& width,
    const int& height,
    const size_t& batch_size,
    const std::string& path
) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        { ".mp4" },
        [width, height] (const std::string& file, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) -> void {
            lifuren::video::feature(width, height, file, labels, features, device);
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

}

#endif // END OF LFR_HEADER_CV_VIDEO_DATASET_HPP
