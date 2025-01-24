/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 视频数据集
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CV_VIDEO_DATASET_HPP
#define LFR_HEADER_CV_VIDEO_DATASET_HPP

#include "lifuren/Dataset.hpp"

namespace lifuren::video {

extern lifuren::dataset::FileDatasetLoader loadFileDatasetLoader(
    const int& width,
    const int& height,
    const size_t& batch_size,
    const std::string& path
);

}

#endif // END OF LFR_HEADER_CV_VIDEO_DATASET_HPP
