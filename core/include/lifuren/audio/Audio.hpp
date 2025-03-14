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

template<typename M>
using AudioModelClientImpl = ModelClientImpl<lifuren::config::ModelParams, std::string, std::string, M>;

template<typename M>
class AudioClient : public AudioModelClientImpl<M> {

public:
    std::tuple<bool, std::string> pred(const std::string& input) override;

};

using AudioModelClient = ModelClient<lifuren::config::ModelParams, std::string, std::string>;

/**
 * @param model 模型名称
 * 
 * @return 模型终端
 */
extern std::unique_ptr<lifuren::audio::AudioModelClient> getAudioClient(const std::string& model);

/**
 * 数据集预处理
 * 
 * @param path 数据集目录
 * 
 * @return 是否成功
 */
extern bool datasetPreprocessingBach(const std::string& path);
extern bool datasetPreprocessingShikuang(const std::string& path);
extern bool datasetPreprocessingBeethoven(const std::string& path);

} // END OF lifuren::audio

#endif // END OF LFR_HEADER_CORE_AUDIO_HPP
