/**
 * 视频数据集
 */
#ifndef LFR_HEADER_CV_VIDEO_DATASET_HPP
#define LFR_HEADER_CV_VIDEO_DATASET_HPP

#include "lifuren/Dataset.hpp"

namespace lifuren::dataset {

namespace video {

extern torch::Tensor feature(const std::string& file, const torch::DeviceType& type);

}

}

#endif // END OF LFR_HEADER_CV_VIDEO_DATASET_HPP
