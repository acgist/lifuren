/**
 * 音频工具
 * 
 * PCM和封装格式都使用单声道
 */
#ifndef LFR_HEADER_CV_AUDIO_HPP
#define LFR_HEADER_CV_AUDIO_HPP

#include <string>

namespace lifuren {

namespace audio {

/**
 * 音频文件转为PCM文件
 * 文件格式：AAC/MP3/FLAC
 * PCM格式：48000Hz mono 16bit
 */
extern bool toPcm(const std::string& audioFile);

/**
 * PCM文件转为音频文件
 */
extern bool toFile(const std::string& pcmFile);

/**
 * 文件夹里面所有音频文件转为PCM文件
 */
extern bool preprocessing(const std::string& path);

} // END OF audio
    
} // END OF lifuren

#endif // END OF LFR_HEADER_CV_AUDIO_HPP
