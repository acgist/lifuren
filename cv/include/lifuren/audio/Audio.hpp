/**
 * 音频工具
 */
#ifndef LFR_HEADER_CV_AUDIO_HPP
#define LFR_HEADER_CV_AUDIO_HPP

#include <string>

namespace lifuren {

namespace audio {

/**
 * 音频文件转为PCM音频
 * 支持格式：AAC/MP3/FLAC
 * PCM格式：单声道 48000采样率 16bits
 */
extern bool toPcm(const std::string& audioFile);

/**
 * PCM转为封装音频
 */
extern bool toFile(const std::string& pcmFile);

} // END OF audio
    
} // END OF lifuren

#endif // END OF LFR_HEADER_CV_AUDIO_HPP
