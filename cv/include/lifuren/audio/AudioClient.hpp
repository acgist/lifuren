/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 音频终端
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CV_AUDIO_CLIENT_HPP
#define LFR_HEADER_CV_AUDIO_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

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

extern std::unique_ptr<lifuren::AudioModelClient> getAudioClient(const std::string& client);

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_AUDIO_CLIENT_HPP
