/**
 * 音频终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_AUDIO_CLIENT_HPP
#define LFR_HEADER_CV_AUDIO_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

/**
 * 音频推理配置
 */
struct AudioParams {

    std::string model;  // 模型路径
    std::string audio;  // 音频文件
    std::string output; // 输出位置
    
};

using AudioModelClient = ModelClient<lifuren::config::ModelParams, AudioParams, std::string>;

template<typename M>
using AudioModelImplClient = ModelImplClient<lifuren::config::ModelParams, AudioParams, std::string, M>;

extern std::unique_ptr<lifuren::AudioModelClient> getAudioClient(const std::string& client);

/**
 * 作曲终端
 */
template<typename M>
class AudioClient : public AudioModelImplClient<M> {

public:
    AudioClient();
    virtual ~AudioClient();

public:
    std::tuple<bool, std::string> pred(const AudioParams& input) override;

};

template<typename M>
using ComposeClient = AudioClient<M>;

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_AUDIO_CLIENT_HPP
