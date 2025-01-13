#include "lifuren/audio/Audio.hpp"

#include <chrono>
#include <fstream>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/audio/AudioDataset.hpp"

#ifndef MONO_NB_CHANNELS
#define MONO_NB_CHANNELS 1
#endif

extern "C" {

#include "libavutil/opt.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswresample/swresample.h"

}

static bool decode(AVFrame* frame, AVPacket* packet, SwrContext** swrCtx, AVCodecContext* decodeCodecCtx, std::vector<char>& buffer, std::ofstream& output);
static bool encode(int64_t& pts, const int64_t& nb_samples, AVFrame* frame, AVPacket* packet, AVCodecContext* encodeCodecCtx, AVFormatContext* outputCtx);
static bool open_input (AVPacket** packet, AVFormatContext** inputCtx, const std::string& file);
static void close_input(AVPacket** packet, AVFormatContext** inputCtx);
static bool open_decoder (AVFrame** frame, AVCodecContext** decodeCodecCtx, const std::string& ext);
static void close_decoder(AVFrame** frame, AVCodecContext** decodeCodecCtx);
static bool open_swr (SwrContext** swrCtx, AVFrame* frame, AVCodecContext* decodeCodecCtx);
static void close_swr(SwrContext** swrCtx);
static bool open_encoder (AVFrame** frame, AVPacket** packet, AVCodecContext** encodeCodecCtx);
static void close_encoder(AVFrame** frame, AVPacket** packet, AVCodecContext** encodeCodecCtx);
static bool open_output (AVFormatContext** outputCtx, AVCodecContext* encodeCodecCtx, int& stream_index, const std::string& file);
static void close_output(AVFormatContext** outputCtx);
static void embedding(const std::string& source, const std::string& target, std::ofstream& stream);
static void embedding(std::ofstream& stream, float norm_factor, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& tuple);

inline static std::string toPcmFilePath(const std::string& audioFile) {
    auto pos = audioFile.find_last_of('.') + 1;
    auto ext = audioFile.substr(pos, audioFile.length() - pos);
    auto pcmFile = audioFile.substr(0, pos) + "pcm";
    return pcmFile;
}

static std::mutex embedding_mutex;

bool lifuren::audio::toPcm(const std::string& audioFile) {
    AVPacket       * packet  { nullptr };
    AVFormatContext* inputCtx{ nullptr };
    if(!open_input(&packet, &inputCtx, audioFile)) {
        close_input(&packet, &inputCtx);
        return false;
    }
    auto pos = audioFile.find_last_of('.') + 1;
    auto ext = audioFile.substr(pos, audioFile.length() - pos);
    auto pcmFile = audioFile.substr(0, pos) + "pcm";
    AVFrame       * frame         { nullptr };
    AVCodecContext* decodeCodecCtx{ nullptr };
    if(!open_decoder(&frame, &decodeCodecCtx, ext)) {
        close_input(&packet, &inputCtx);
        close_decoder(&frame, &decodeCodecCtx);
        return false;
    }
    std::ofstream output;
    output.open(pcmFile, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if(!output.is_open()) {
        output.close();
        close_input(&packet, &inputCtx);
        close_decoder(&frame, &decodeCodecCtx);
        SPDLOG_WARN("打开音频输出文件失败：{}", pcmFile);
        return false;
    }
    std::vector<char> buffer;
    buffer.resize(16 * 1024);
    SwrContext* swrCtx{ nullptr };
    while(av_read_frame(inputCtx, packet) == 0) {
        if(!decode(frame, packet, &swrCtx, decodeCodecCtx, buffer, output)) {
            break;
        }
    }
    // 刷出缓冲
    decode(frame, NULL, &swrCtx, decodeCodecCtx, buffer, output);
    // 释放资源
    output.close();
    close_swr(&swrCtx);
    close_input(&packet, &inputCtx);
    close_decoder(&frame, &decodeCodecCtx);
    return true;
}

bool lifuren::audio::toFile(const std::string& pcmFile) {
    AVFrame       * frame { nullptr };
    AVPacket      * packet{ nullptr };
    AVCodecContext* encodeCodecCtx{ nullptr };
    if(!open_encoder(&frame, &packet, &encodeCodecCtx)) {
        close_encoder(&frame, &packet, &encodeCodecCtx);
        return false;
    }
    int stream_index;
    AVFormatContext* outputCtx{ nullptr };
    auto pos = pcmFile.find_last_of('.') + 1;
    #ifdef __MP3__
    auto outputFile = pcmFile.substr(0, pos) + "mp3";
    #else
    auto outputFile = pcmFile.substr(0, pos) + "flac";
    #endif
    if(!open_output(&outputCtx, encodeCodecCtx, stream_index, outputFile)) {
        close_encoder(&frame, &packet, &encodeCodecCtx);
        close_output(&outputCtx);
        return false;
    }
    const int sample_size = av_get_bytes_per_sample(AV_SAMPLE_FMT_S16);
    #ifdef __MP3__
    const int buffer_size = av_samples_get_buffer_size(NULL, MONO_NB_CHANNELS, 1152, AV_SAMPLE_FMT_S16, 0);
    #else
    const int buffer_size = av_samples_get_buffer_size(NULL, MONO_NB_CHANNELS, 4608, AV_SAMPLE_FMT_S16, 0);
    #endif
    int64_t pts = 0;
    int64_t size;
    int64_t nb_samples;
    std::vector<char> data;
    data.resize(buffer_size);
    std::ifstream input;
    input.open(pcmFile, std::ios_base::in | std::ios_base::binary);
    if(!input.is_open()) {
        input.close();
        close_encoder(&frame, &packet, &encodeCodecCtx);
        close_output(&outputCtx);
        SPDLOG_WARN("打开音频输入文件失败：{}", pcmFile);
        return false;
    }
    // 写入头部
    if(avformat_write_header(outputCtx, NULL) != 0) {
        SPDLOG_WARN("写入头部失败");
    }
    while(input.read(data.data(), buffer_size)) {
        size       = input.gcount();
        nb_samples = size / sample_size;
        frame->pts            = AV_NOPTS_VALUE;
        frame->format         = AV_SAMPLE_FMT_S16;
        #if !defined(FF_API_OLD_CHANNEL_LAYOUT) || !FF_API_OLD_CHANNEL_LAYOUT
        frame->ch_layout      = AV_CHANNEL_LAYOUT_MONO;
        #else
        frame->channel_layout = AV_CH_LAYOUT_MONO;
        #endif
        frame->nb_samples     = nb_samples;
        frame->sample_rate    = 48000;
        // 申请空间
        if(av_frame_get_buffer(frame, 0) != 0) {
            av_frame_unref(frame);
            break;
        }
        std::memcpy(frame->buf[0]->data, data.data(), size);
        if(!encode(pts, nb_samples, frame, packet, encodeCodecCtx, outputCtx)) {
            break;
        }
    }
    // 刷出缓冲
    encode(pts, nb_samples, NULL, packet, encodeCodecCtx, outputCtx);
    // 写入尾部
    if(av_write_trailer(outputCtx)) {
        SPDLOG_WARN("写入尾部失败");
    }
    // 释放资源
    input.close();
    close_encoder(&frame, &packet, &encodeCodecCtx);
    close_output(&outputCtx);
    return true;
}

bool lifuren::audio::embedding(const std::string& path, const std::string& dataset, std::ofstream& stream, lifuren::thread::ThreadPool& pool) {
    const std::string source = "source";
    const std::string target = "target";
    std::vector<std::string> files;
    lifuren::file::listFile(files, dataset, { ".aac", ".mp3", ".flac" });
    if(files.empty()) {
        SPDLOG_DEBUG("音频嵌入文件为空");
        return true;
    }
    for(const auto& file : files) {
        const auto index = file.find_last_of('.');
        if(index == std::string::npos) {
            SPDLOG_INFO("加载文件匹配规则失败：{}", file);
            continue;
        }
        if(index < source.size()) {
            SPDLOG_INFO("加载文件匹配规则失败：{}", file);
            continue;
        }
        const auto label = file.substr(index - source.size(), source.size());
        if(label != source) {
            if(label != target) {
                SPDLOG_INFO("加载文件匹配规则失败：{}", file);
            }
            continue;
        }
        auto target_file(file);
        target_file.replace(index - source.size(), source.size(), target);
        const auto iterator = std::find(files.begin(), files.end(), target_file);
        if(iterator == files.end()) {
            SPDLOG_INFO("加载文件没有标记文件：{}", file);
            continue;
        }
        pool.submit([&stream, file, target_file]() {
            SPDLOG_DEBUG("加载文件：{} - {}", file, target_file);
            if(lifuren::audio::toPcm(file) && lifuren::audio::toPcm(target_file)) {
                SPDLOG_DEBUG("转换PCM成功：{} - {}", file, target_file);
                ::embedding(toPcmFilePath(file), toPcmFilePath(target_file), stream);
            } else {
                SPDLOG_WARN("转换PCM失败：{} - {}", file, target_file);
            }
        });
    }
    return true;
}

static bool decode(AVFrame* frame, AVPacket* packet, SwrContext** swrCtx, AVCodecContext* decodeCodecCtx, std::vector<char>& buffer, std::ofstream& output) {
    if(avcodec_send_packet(decodeCodecCtx, packet) != 0) {
        av_packet_unref(packet);
        return false;
    }
    if(packet) {
        av_packet_unref(packet);
    }
    while(avcodec_receive_frame(decodeCodecCtx, frame) == 0) {
        if(!*swrCtx && !open_swr(swrCtx, frame, decodeCodecCtx)) {
            av_frame_unref(frame);
            return false;
        }
        uint8_t* buffer_data = reinterpret_cast<uint8_t*>(buffer.data());
        const int swr_size = swr_convert(
            *swrCtx,
            &buffer_data,
            frame->nb_samples,
            const_cast<const uint8_t**>(frame->data),
            frame->nb_samples
        );
        av_frame_unref(frame);
        const int buffer_size = av_samples_get_buffer_size(NULL, MONO_NB_CHANNELS, swr_size, AV_SAMPLE_FMT_S16, 0);
        output.write(buffer.data(), buffer_size);
    }
    av_frame_unref(frame);
    return true;
}

static bool encode(int64_t& pts, const int64_t& nb_samples, AVFrame* frame, AVPacket* packet, AVCodecContext* encodeCodecCtx, AVFormatContext* outputCtx) {
    if(avcodec_send_frame(encodeCodecCtx, frame) != 0) {
        av_frame_unref(frame);
        return false;
    }
    if(frame) {
        av_frame_unref(frame);
    }
    while(avcodec_receive_packet(encodeCodecCtx, packet) == 0) {
        packet->pts = pts;
        packet->dts = pts;
        pts += nb_samples;
        av_write_frame(outputCtx, packet);
        av_packet_unref(packet);
    }
    av_packet_unref(packet);
    return true;
}

static bool open_input(AVPacket** packet, AVFormatContext** inputCtx, const std::string& file) {
    *packet = av_packet_alloc();
    if(!*packet) {
        return false;
    }
    *inputCtx = avformat_alloc_context();
    if(!*inputCtx) {
        return false;
    }
    if(avformat_open_input(inputCtx, file.c_str(), NULL, NULL) != 0) {
        SPDLOG_WARN("打开音频文件失败：{}", file);
        return false;
    }
    return true;
}

static void close_input(AVPacket** packet, AVFormatContext** inputCtx) {
    if(*packet) {
        av_packet_free(packet);
        *packet = nullptr;
    }
    if(*inputCtx) {
        avformat_close_input(inputCtx);
        *inputCtx = nullptr;
    }
}

static bool open_decoder(AVFrame** frame, AVCodecContext** decodeCodecCtx, const std::string& ext) {
    *frame = av_frame_alloc();
    if(!*frame) {
        return false;
    }
    const AVCodec* decoder;
    if("aac" == ext || "AAC" == ext) {
        decoder = avcodec_find_decoder(AV_CODEC_ID_AAC);
    } else if("mp3" == ext || "MP3" == ext) {
        decoder = avcodec_find_decoder(AV_CODEC_ID_MP3);
    } else if("flac" == ext || "FLAC" == ext) {
        decoder = avcodec_find_decoder(AV_CODEC_ID_FLAC);
    } else {
        return false;
    }
    if(!decoder) {
        SPDLOG_WARN("不支持的解码格式：{}", ext);
        return false;
    }
    *decodeCodecCtx = avcodec_alloc_context3(decoder);
    if(!*decodeCodecCtx) {
        SPDLOG_WARN("创建解码器上下文失败：{}", ext);
        return false;
    }
    if(avcodec_open2(*decodeCodecCtx, decoder, nullptr) != 0) {
        SPDLOG_WARN("打开解码器上下文失败：{}", ext);
        return false;
    }
    return true;
}

static void close_decoder(AVFrame** frame, AVCodecContext** decodeCodecCtx) {
    if(*frame) {
        av_frame_free(frame);
        *frame = nullptr;
    }
    if(*decodeCodecCtx) {
        avcodec_free_context(decodeCodecCtx);
        *decodeCodecCtx = nullptr;
    }
}

static bool open_swr(SwrContext** swrCtx, AVFrame* frame, AVCodecContext* decodeCodecCtx) {
    #if !defined(FF_API_OLD_CHANNEL_LAYOUT) || !FF_API_OLD_CHANNEL_LAYOUT
    static AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
    #endif
    *swrCtx = swr_alloc();
    av_opt_set_int       (*swrCtx, "in_sample_rate",     frame->sample_rate,         0);
    av_opt_set_int       (*swrCtx, "out_sample_rate",    48000,                      0);
    av_opt_set_sample_fmt(*swrCtx, "in_sample_fmt",      decodeCodecCtx->sample_fmt, 0);
    av_opt_set_sample_fmt(*swrCtx, "out_sample_fmt",     AV_SAMPLE_FMT_S16,          0);
    #if !defined(FF_API_OLD_CHANNEL_LAYOUT) || !FF_API_OLD_CHANNEL_LAYOUT
    av_opt_set_chlayout  (*swrCtx, "in_channel_layout",  &frame->ch_layout,          0);
    av_opt_set_chlayout  (*swrCtx, "out_channel_layout", &mono,                      0);
    #else
    av_opt_set_channel_layout(*swrCtx, "in_channel_layout",  frame->channel_layout,  0);
    av_opt_set_channel_layout(*swrCtx, "out_channel_layout", AV_CH_LAYOUT_MONO,      0);
    #endif
    if(swr_init(*swrCtx) != 0) {
        SPDLOG_WARN("初始化重采样失败");
        return false;
    }
    return true;
}

static void close_swr(SwrContext** swrCtx) {
    if(*swrCtx) {
        swr_free(swrCtx);
        *swrCtx = nullptr;
    }
}

static bool open_encoder(AVFrame** frame, AVPacket** packet, AVCodecContext** encodeCodecCtx) {
    *frame = av_frame_alloc();
    if(!*frame) {
        return false;
    }
    *packet = av_packet_alloc();
    if(!*packet) {
        return false;
    }
    #ifdef __MP3__
    const AVCodec* encoder = avcodec_find_encoder(AV_CODEC_ID_MP3);
    #else
    const AVCodec* encoder = avcodec_find_encoder(AV_CODEC_ID_FLAC);
    #endif
    if(!encoder) {
        SPDLOG_WARN("不支持的解码格式");
        return false;
    }
    *encodeCodecCtx = avcodec_alloc_context3(encoder);
    if(!*encodeCodecCtx) {
        SPDLOG_WARN("创建编码器上下文失败");
        return false;
    }
    #if defined(__MP3__) && (defined(__linux) || defined(__linux__))
    (*encodeCodecCtx)->sample_fmt     = AV_SAMPLE_FMT_S16P;
    #else
    (*encodeCodecCtx)->sample_fmt     = AV_SAMPLE_FMT_S16;
    #endif
    #if !defined(FF_API_OLD_CHANNEL_LAYOUT) || !FF_API_OLD_CHANNEL_LAYOUT
    (*encodeCodecCtx)->ch_layout      = AV_CHANNEL_LAYOUT_MONO;
    #else
    (*encodeCodecCtx)->channel_layout = AV_CH_LAYOUT_MONO;
    #endif
    (*encodeCodecCtx)->sample_rate    = 48000;
    if(avcodec_open2(*encodeCodecCtx, encoder, nullptr) != 0) {
        SPDLOG_WARN("打开编码器上下文失败");
        return false;
    }
    return true;
}

static void close_encoder(AVFrame** frame, AVPacket** packet, AVCodecContext** encodeCodecCtx) {
    if(*frame) {
        av_frame_free(frame);
        *frame = nullptr;
    }
    if(*packet) {
        av_packet_free(packet);
        *packet = nullptr;
    }
    if(*encodeCodecCtx) {
        avcodec_free_context(encodeCodecCtx);
        *encodeCodecCtx = nullptr;
    }
}

static bool open_output(AVFormatContext** outputCtx, AVCodecContext* encodeCodecCtx, int& stream_index, const std::string& file) {
    *outputCtx = avformat_alloc_context();
    if(!*outputCtx) {
        return false;
    }
    if(avformat_alloc_output_context2(outputCtx, NULL, NULL, file.c_str()) != 0) {
        SPDLOG_WARN("打开音频输出文件失败：{}", file);
        return false;
    }
    #ifdef __MP3__
    const AVCodec* encoder = avcodec_find_encoder(AV_CODEC_ID_MP3);
    #else
    const AVCodec* encoder = avcodec_find_encoder(AV_CODEC_ID_FLAC);
    #endif
    if(!encoder) {
        SPDLOG_WARN("不支持的解码格式：{}", file);
        return false;
    }
    AVStream* stream = avformat_new_stream(*outputCtx, encoder);
    if(!stream) {
        SPDLOG_WARN("打开音频流失败：{}", file);
        return false;
    }
    if(avcodec_parameters_from_context(stream->codecpar, encodeCodecCtx) != 0) {
        SPDLOG_WARN("拷贝音频配置失败：{}", file);
        return false;
    }
    if(avio_open(&(*outputCtx)->pb, file.c_str(), AVIO_FLAG_WRITE) != 0) {
        SPDLOG_WARN("打开音频输出文件失败：{}", file);
        return false;
    }
    stream_index = stream->index;
    return true;
}

static void close_output(AVFormatContext** outputCtx) {
    if(*outputCtx) {
        avformat_close_input(outputCtx);
        *outputCtx = nullptr;
    }
}

static void embedding(const std::string& source, const std::string& target, std::ofstream& stream) {
    std::ifstream source_stream;
    std::ifstream target_stream;
    source_stream.open(source, std::ios_base::binary);
    target_stream.open(target, std::ios_base::binary);
    if(!source_stream.is_open() || !target_stream.is_open()) {
        SPDLOG_DEBUG("打开文件失败：{} - {}", source, target);
        source_stream.close();
        target_stream.close();
        return;
    }
    std::vector<short> source_pcm;
    std::vector<short> target_pcm;
    SPDLOG_DEBUG("开始音频嵌入：{} - {}", source, target);
    source_pcm.resize(DATASET_PCM_LENGTH);
    target_pcm.resize(DATASET_PCM_LENGTH);
    while(
        source_stream.read(reinterpret_cast<char*>(source_pcm.data()), DATASET_PCM_LENGTH * sizeof(short)) &&
        target_stream.read(reinterpret_cast<char*>(target_pcm.data()), DATASET_PCM_LENGTH * sizeof(short)) &&
        source_stream.gcount() == target_stream.gcount()
    ) {
        float norm_factor;
        // 短时傅里叶变换
        auto source_tuple = lifuren::dataset::audio::pcm_mag_pha_stft(source_pcm, norm_factor);
        embedding(stream, norm_factor, source_tuple);
        auto target_tuple = lifuren::dataset::audio::pcm_mag_pha_stft(target_pcm, norm_factor);
        embedding(stream, norm_factor, target_tuple);
    }
    source_stream.close();
    target_stream.close();
    stream.flush();
    SPDLOG_DEBUG("音频嵌入完成：{} - {}", source, target);
}

inline static void embedding(std::ofstream& stream, float norm_factor, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& tuple) {
    auto tensor = torch::stack({ std::get<0>(tuple), std::get<1>(tuple) }).squeeze();
    {
        std::lock_guard<std::mutex> lock(embedding_mutex);
        stream.write(reinterpret_cast<char*>(&norm_factor), sizeof(norm_factor));
        stream.write(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel() * tensor.element_size());
    }
}
