/**
 * 图片数据集
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_IMAGE_DATASET_HPP
#define LFR_HEADER_CV_IMAGE_DATASET_HPP

#include <cstdint>
#include <cstdlib>

#include "lifuren/Dataset.hpp"
#include "lifuren/image/Image.hpp"

namespace lifuren::dataset {

/**
 * @param width      图片宽度
 * @param height     图片高度
 * @param batch_size 批次大小
 * @param path       图片路径
 * @param image_type 图片格式
 * @param mapping    标签映射
 * @param transform  图片转换
 * 
 * @return 图片数据集
 */
inline auto loadImageFileDataset(
    const int& width,
    const int& height,
    const size_t batch_size,
    const std::string& path,
    const std::string& image_type,
    const std::map<std::string, float>& classify,
    const std::function<void(const cv::Mat&)> transform = nullptr
) -> decltype(auto) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        { image_type },
        classify,
        [width, height, transform] (const std::string& file) -> torch::Tensor {
            size_t length{ 0 };
            std::vector<float> feature;
            feature.resize(width * height * 3);
            lifuren::image::load(file, feature.data(), length, width, height, transform);
            return torch::from_blob(feature.data(), { height, width, 3 }, torch::kByte).permute({2, 0, 1}).to(torch::kF32).div(255.0);
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using ImageFileDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadImageFileDataset),
    const int&,
    const int&,
    const size_t&,
    const std::string&,
    const std::string&,
    const std::map<std::string, float>&,
    const std::function<void(const cv::Mat&)>
>::type;

} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_CV_IMAGE_DATASET_HPP
