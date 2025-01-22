/**
 * 图片数据集
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_IMAGE_DATASET_HPP
#define LFR_HEADER_CV_IMAGE_DATASET_HPP

#include "lifuren/Dataset.hpp"

namespace lifuren::dataset {

namespace image {

extern torch::Tensor feature(const int& width, const int& height, const std::string& file, const torch::DeviceType& type);

inline torch::Tensor feature(char* data, const int& width, const int& height, const torch::DeviceType& type) {
    return torch::from_blob(data, { height, width, 3 }, torch::kByte).permute({2, 0, 1}).to(torch::kFloat32).div(255.0).clone().to(type);
}

} // END OF image

/**
 * @param width      图片宽度
 * @param height     图片高度
 * @param batch_size 批次大小
 * @param path       图片路径
 * @param classify   标签映射
 * 
 * @return 图片数据集
 */
inline FileDatasetLoader loadImageFileClassifyDataset(
    const int& width,
    const int& height,
    const size_t batch_size,
    const std::string& path,
    const std::map<std::string, float>& classify
) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        { ".jpg", ".png", ".jpeg" },
        classify,
        [width, height] (const std::string& file, const torch::DeviceType& device) -> torch::Tensor {
            return lifuren::dataset::image::feature(width, height, file, device);
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_CV_IMAGE_DATASET_HPP
