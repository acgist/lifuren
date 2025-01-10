/**
 * 音频工具
 */
#ifndef LFR_HEADER_CV_AUDIO_HPP
#define LFR_HEADER_CV_AUDIO_HPP

#include <string>

#include "lifuren/Thread.hpp"

namespace lifuren {
namespace audio   {

/**
 * 音频文件转为PCM文件
 * 文件格式：AAC/MP3/FLAC
 * PCM格式：48000Hz mono 16bit
 * 
 * @param audioFile 音频文件
 * 
 * @return 是否成功
 */
extern bool toPcm(const std::string& audioFile);

/**
 * PCM文件转为音频文件
 */
extern bool toFile(const std::string& pcmFile);

/**
 * 音频嵌入
 * 1. PCM
 * 2. STFT
 */
extern bool embedding(const std::string& path, const std::string& dataset, std::ofstream& stream, lifuren::thread::ThreadPool& pool);

} // END OF audio
} // END OF lifuren

#endif // END OF LFR_HEADER_CV_AUDIO_HPP
