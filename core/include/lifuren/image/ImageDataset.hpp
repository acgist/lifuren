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

#include "lifuren/Dataset.hpp"

namespace cv {

    class Mat;

} // END OF cv

namespace lifuren::image {

/**
 * 修改图片大小
 */
extern void resize(
    cv::Mat& image,  // 图片数据
    const int width, // 目标宽度
    const int height // 目标高度
);

/**
 * 图片转为张量
 * 
 * @return 图片张量
 */
extern torch::Tensor feature(
    const cv::Mat& image, // 图片数据
    const int width,      // 图片宽度
    const int height,     // 图片高度
    const torch::DeviceType& type // 设备类型
);

/**
 * 张量转为图片
 */
extern void tensor_to_mat(
    cv::Mat& mat, // 图片数据：需要提前申请空间
    const torch::Tensor& tensor // 图片张量
);

/**
 * @return 图片数据集
 */
extern lifuren::dataset::FileDatasetLoader loadFileDatasetLoader(
    const int width,  // 图片宽度
    const int height, // 图片高度
    const size_t batch_size, // 批量大小
    const std::string& path, // 数据集路径
    const std::map<std::string, float>& classify // 图片分类
);

} // END OF lifuren::image

#endif // END OF LFR_HEADER_CV_IMAGE_DATASET_HPP
