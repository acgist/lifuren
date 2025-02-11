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

/**
 * @return 视频数据集
 */
extern lifuren::dataset::FileDatasetLoader loadFileDatasetLoader(
    const int width,  // 视频宽度
    const int height, // 视频高度
    const size_t batch_size, // 批量大小
    const std::string& path  // 数据集路径
);

} // END OF lifuren::video

#endif // END OF LFR_HEADER_CV_VIDEO_DATASET_HPP
