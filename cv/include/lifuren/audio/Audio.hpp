/**
 * Copyright(c) 2024-present acgist. ALl Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 音频工具
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CV_AUDIO_HPP
#define LFR_HEADER_CV_AUDIO_HPP

#include <string>

#include "lifuren/Thread.hpp"

namespace lifuren::audio {

/**
 * 音频文件转为PCM文件
 * 
 * 支持音频文件格式：AAC/MP3/FLAC
 * 
 * PCM文件格式：48000Hz mono 16bit
 * 
 * @return <是否成功, PCM文件路径>
 */
extern std::tuple<bool, std::string> toPcm(
    const std::string& audioFile // 音频文件
);

/**
 * PCM文件转为音频文件
 * 
 * PCM文件格式：48000Hz mono 16bit
 * 
 * @return <是否成功, 音频文件路径>
 */
extern std::tuple<bool, std::string> toFile(
    const std::string& pcmFile // PCM文件
);

/**
 * 音频嵌入
 * 
 * @return 是否成功
 */
extern bool embedding(
    const std::string& path,    // 数据集上级目录
    const std::string& dataset, // 数据集目录
    std::ofstream    & stream,  // 嵌入文件流
    lifuren::thread::ThreadPool& pool // 线程池
);

} // END OF lifuren::audio

#endif // END OF LFR_HEADER_CV_AUDIO_HPP
