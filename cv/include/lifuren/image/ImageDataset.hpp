/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 图片数据集
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CV_IMAGE_DATASET_HPP
#define LFR_HEADER_CV_IMAGE_DATASET_HPP

#include <string>

#include "lifuren/Dataset.hpp"

namespace cv {

    class Mat;

} // END OF cv

namespace lifuren::image {

/**
 * @return 是否成功
 */
extern bool read(
    const std::string& path, // 图片路径
    char* data,              // 图片数据
    const size_t width,      // 图片宽度
    const size_t height      // 图片高度
);

extern bool read(
    cv::Mat& image,     // 图片原始数据
    char* data,         // 图片目标数据
    const size_t width, // 图片高度
    const size_t height // 图片宽度
);

/**
 * @return 是否成功
 */
extern bool write(
    const std::string& path, // 图片路径
    const char* data,        // 图片数据
    const size_t width,      // 图片宽度
    const size_t height      // 图片高度
);

/**
 * @return 图片张量
 */
inline torch::Tensor feature(
    char* data,                   // 图片数据
    const int width,              // 图片宽度
    const int height,             // 图片高度
    const torch::DeviceType& type // 设备类型
) {
    return torch::from_blob(data, { height, width, 3 }, torch::kByte).permute({2, 0, 1}).to(torch::kFloat32).div(255.0).clone().to(type);
}

/**
 * @return 图片数据集
 */
extern lifuren::dataset::FileDatasetLoader loadFileDatasetLoader(
    const int width,         // 图片宽度
    const int height,        // 图片高度
    const size_t batch_size, // 批量大小
    const std::string& path, // 数据集路径
    const std::map<std::string, float>& classify // 图片分类
);

} // END OF lifuren::image

#endif // END OF LFR_HEADER_CV_IMAGE_DATASET_HPP
