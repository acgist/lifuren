/**
 * 视频数据集
 */
#ifndef LFR_HEADER_CV_VIDEO_DATASET_HPP
#define LFR_HEADER_CV_VIDEO_DATASET_HPP

#include "lifuren/Dataset.hpp"

namespace lifuren::dataset {

namespace video {

}

inline auto loadVideoFileGANDataset(
    const size_t& batch_size,
    const std::string& path
) -> decltype(auto) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        { ".mp4" },
        [] (const std::string& file, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) -> void {
            // TODO: 时间切片
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using VideoFileGANDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadVideoFileGANDataset),
    const size_t&,
    const std::string&
>::type;

inline auto loadVideoFileStyleDataset(
    const size_t& batch_size,
    const std::string& path
) -> decltype(auto) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        "source",
        "target",
        { ".mp4" },
        [] (const std::string& source, const std::string& target, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) -> void {
            // TODO: 视频切片
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using VideoFileStyleDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadVideoFileStyleDataset),
    const size_t&,
    const std::string&
>::type;

}

#endif // END OF LFR_HEADER_CV_VIDEO_DATASET_HPP
