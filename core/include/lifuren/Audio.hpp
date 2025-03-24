/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 音频
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_AUDIO_HPP
#define LFR_HEADER_CORE_AUDIO_HPP

#include "lifuren/Client.hpp"

namespace lifuren::audio {

using AudioModelClient = ModelClient<lifuren::config::ModelParams, std::string, std::string>;

/**
 * @param model 模型名称
 * 
 * @return 模型终端
 */
extern std::unique_ptr<lifuren::audio::AudioModelClient> getAudioClient(const std::string& model);

/**
 * 师旷数据集处理
 * 
 * @param path 数据集目录
 * 
 * @return 是否成功
 */
extern bool allDatasetPreprocessShikuang(const std::string& path);

} // END OF lifuren::audio

#endif // END OF LFR_HEADER_CORE_AUDIO_HPP
