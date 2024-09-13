/**
 * 图片数据集
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_IMAGE_DATASETS_HPP
#define LFR_HEADER_CV_IMAGE_DATASETS_HPP

#include <cstdint>
#include <cstdlib>

#include "lifuren/Images.hpp"
#include "lifuren/Datasets.hpp"

namespace lifuren {

/**
 * 图片数据集
 * 
 * @param width      图片宽度
 * @param height     图片高度
 * @param batch_size 批次大小
 * @param path       图片路径
 * @param image_type 图片格式
 * @param mapping    标签映射
 * @param transform  图片转换
 */
inline auto loadImageFileDataset(
    const int& width,
    const int& height,
    const size_t& batch_size,
    const std::string& path,
    const std::string& image_type,
    const std::map<std::string, float>& mapping,
    const std::function<void(const cv::Mat&)> transform = nullptr
) -> decltype(auto) {
    auto dataset = lifuren::datasets::FileDataset(
        batch_size,
        path,
        { image_type },
        [width, height, transform](const std::string& path, std::vector<std::vector<float>>& features) {
            std::vector<float> feature;
            feature.resize(width * height * 3);
            size_t length{ 0 };
            lifuren::images::readTransform(path, feature.data(), length, width, height, transform);
            if(length > 0LL) {
                features.push_back(std::move(feature));
            }
        },
        mapping
    );
    return dataset;
}

using ImageFileDatasetLoader = std::invoke_result<
    decltype(&lifuren::loadImageFileDataset),
    const int&,
    const int&,
    const size_t&,
    const std::string&,
    const std::string&,
    const std::map<std::string, float>&,
    const std::function<void(const cv::Mat&)>
>::type;

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_IMAGE_DATASETS_HPP
