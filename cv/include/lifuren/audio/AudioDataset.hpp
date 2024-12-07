/**
 * 音频数据集
 */
#ifndef LFR_HEADER_CV_AUDIO_DATASET_HPP
#define LFR_HEADER_CV_AUDIO_DATASET_HPP

#include "lifuren/Dataset.hpp"

namespace lifuren::dataset {

namespace audio {

extern torch::Tensor feature(const std::string& file, const torch::DeviceType& type);

}
    
} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_CV_AUDIO_DATASET_HPP
