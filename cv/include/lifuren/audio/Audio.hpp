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
#ifndef LFR_HEADER_CV_AUDIO_HPP
#define LFR_HEADER_CV_AUDIO_HPP

#include "lifuren/Client.hpp"

namespace lifuren::audio {

/**
 * 音频推理配置
 */
struct AudioParams {

    std::string model;  // 模型文件
    std::string audio;  // 音频文件
    std::string output; // 输出文件
    
};

using AudioModelClient = ModelClient<lifuren::config::ModelParams, AudioParams, std::string>;

template<typename M>
using AudioModelImplClient = ModelImplClient<lifuren::config::ModelParams, AudioParams, std::string, M>;

/**
 * 作曲终端
 */
template<typename M>
class AudioClient : public AudioModelImplClient<M> {

public:
    std::tuple<bool, std::string> pred(const AudioParams& input) override;

};

template<typename M>
using ComposeClient = AudioClient<M>;

/**
 * @return 作曲终端
 */
extern std::unique_ptr<lifuren::audio::AudioModelClient> getAudioClient(
    const std::string& model // 模型名称
);

/**
 * 数据集预处理
 * 
 * @return 是否成功
 */
extern bool datasetPreprocessing(
    const std::string& path // 数据集目录
);

} // END OF lifuren::audio

#endif // END OF LFR_HEADER_CV_AUDIO_HPP
