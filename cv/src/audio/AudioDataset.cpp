#include "lifuren/audio/AudioDataset.hpp"

#include <chrono>
#include <fstream>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/audio/Audio.hpp"

extern "C" {

#include "libavutil/opt.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswresample/swresample.h"

}

#ifndef MONO_NB_CHANNELS
#define MONO_NB_CHANNELS 1
#endif

// 解码
static bool decode(AVFrame* frame, AVPacket* packet, SwrContext** swrCtx, AVCodecContext* decodeCodecCtx, std::vector<char>& buffer, std::ofstream& output);
static bool open_swr (SwrContext** swrCtx, AVFrame* frame, AVCodecContext* decodeCodecCtx);
static void close_swr(SwrContext** swrCtx);
static bool open_input (AVPacket** packet, AVFormatContext** inputCtx, const std::string& file);
static void close_input(AVPacket** packet, AVFormatContext** inputCtx);
static bool open_decoder (AVFrame** frame, AVCodecContext** decodeCodecCtx, const std::string& suffix);
static void close_decoder(AVFrame** frame, AVCodecContext** decodeCodecCtx);

// 编码
static bool encode(int64_t& pts, const int64_t& nb_samples, AVFrame* frame, AVPacket* packet, AVCodecContext* encodeCodecCtx, AVFormatContext* outputCtx);
static bool open_output (AVFormatContext** outputCtx, AVCodecContext* encodeCodecCtx, int& stream_index, const std::string& file);
static void close_output(AVFormatContext** outputCtx);
static bool open_encoder (AVFrame** frame, AVPacket** packet, AVCodecContext** encodeCodecCtx);
static void close_encoder(AVFrame** frame, AVPacket** packet, AVCodecContext** encodeCodecCtx);

// 嵌入
static void embedding(const std::string& source, const std::string& target, std::ofstream& stream);
static void embedding(std::ofstream& stream, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& tuple);

/**
 * 短时傅里叶变换
 * 
 * 480 mag sizes = pha sizes = [1, 201, 5]
 * 
 * @return [mag, pha, com]
 */
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mag_pha_stft(
    torch::Tensor pcm_norm, // 一维时间序列或二维时间序列批次
    int n_fft    = 400,     // 傅里叶变换的大小
    int hop_size = 100,     // 相邻滑动窗口帧之间的距离：floor(n_fft / 4)
    int win_size = 400,     // 窗口帧和STFT滤波器的大小：n_fft
    float compress_factor = 1.0 // 压缩因子
);

/**
 * 短时傅里叶逆变换
 * 
 * @return PCM
 */
static torch::Tensor mag_pha_istft(
    torch::Tensor mag,  // mag
    torch::Tensor pha,  // pha
    int n_fft    = 400, // 傅里叶变换的大小
    int hop_size = 100, // 相邻滑动窗口帧之间的距离：floor(n_fft / 4)
    int win_size = 400, // 窗口帧和STFT滤波器的大小：n_fft
    float compress_factor = 1.0 // 压缩因子
);

static std::mutex embedding_mutex;

const static float NORMALIZATION = 32768.0F;

std::tuple<bool, std::string> lifuren::audio::toPcm(const std::string& audioFile) {
    AVPacket       * packet  { nullptr };
    AVFormatContext* inputCtx{ nullptr };
    if(!open_input(&packet, &inputCtx, audioFile)) {
        close_input(&packet, &inputCtx);
        return {false, {}};
    }
    const auto pos     = audioFile.find_last_of('.') + 1;
    const auto suffix  = audioFile.substr(pos, audioFile.length() - pos);
    const auto pcmFile = audioFile.substr(0, pos) + "pcm";
    AVFrame       * frame         { nullptr };
    AVCodecContext* decodeCodecCtx{ nullptr };
    if(!open_decoder(&frame, &decodeCodecCtx, suffix)) {
        close_input(&packet, &inputCtx);
        close_decoder(&frame, &decodeCodecCtx);
        return {false, pcmFile};
    }
    std::ofstream output;
    output.open(pcmFile, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if(!output.is_open()) {
        SPDLOG_WARN("打开音频输出文件失败：{}", pcmFile);
        close_input(&packet, &inputCtx);
        close_decoder(&frame, &decodeCodecCtx);
        return {false, pcmFile};
    }
    std::vector<char> buffer;
    buffer.resize(16 * 1024);
    SwrContext* swrCtx{ nullptr };
    while(av_read_frame(inputCtx, packet) == 0) {
        if(!decode(frame, packet, &swrCtx, decodeCodecCtx, buffer, output)) {
            av_packet_unref(packet);
            break;
        }
        av_packet_unref(packet);
    }
    decode(frame, NULL, &swrCtx, decodeCodecCtx, buffer, output);
    output.close();
    close_swr(&swrCtx);
    close_input(&packet, &inputCtx);
    close_decoder(&frame, &decodeCodecCtx);
    return {true, pcmFile};
}

static bool open_input(AVPacket** packet, AVFormatContext** inputCtx, const std::string& file) {
    *packet = av_packet_alloc();
    if(!*packet) {
        SPDLOG_WARN("申请数据包失败：{}", file);
        return false;
    }
    *inputCtx = avformat_alloc_context();
    if(!*inputCtx) {
        SPDLOG_WARN("申请格式上下文失败：{}", file);
        return false;
    }
    if(avformat_open_input(inputCtx, file.c_str(), NULL, NULL) != 0) {
        SPDLOG_WARN("打开格式上下文失败：{}", file);
        return false;
    }
    if((*inputCtx)->nb_streams != 1) {
        SPDLOG_WARN("文件包含多个媒体轨道：{}", file);
        return false;
    }
    if((*inputCtx)->streams[0]->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) {
        SPDLOG_WARN("文件没有音频轨道：{}", file);
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

static bool open_decoder(AVFrame** frame, AVCodecContext** decodeCodecCtx, const std::string& suffix) {
    *frame = av_frame_alloc();
    if(!*frame) {
        SPDLOG_WARN("申请数据帧失败：{}", suffix);
        return false;
    }
    const AVCodec* decoder;
    if("aac" == suffix || "AAC" == suffix) {
        decoder = avcodec_find_decoder(AV_CODEC_ID_AAC);
    } else if("mp3" == suffix || "MP3" == suffix) {
        decoder = avcodec_find_decoder(AV_CODEC_ID_MP3);
    } else if("flac" == suffix || "FLAC" == suffix) {
        decoder = avcodec_find_decoder(AV_CODEC_ID_FLAC);
    } else {
        SPDLOG_WARN("不支持的音频格式：{}", suffix);
        return false;
    }
    if(!decoder) {
        SPDLOG_WARN("不支持的音频格式：{}", suffix);
        return false;
    }
    *decodeCodecCtx = avcodec_alloc_context3(decoder);
    if(!*decodeCodecCtx) {
        SPDLOG_WARN("申请解码器上下文失败：{}", suffix);
        return false;
    }
    if(avcodec_open2(*decodeCodecCtx, decoder, nullptr) != 0) {
        SPDLOG_WARN("打开解码器上下文失败：{}", suffix);
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
    *swrCtx = swr_alloc();
    if(!*swrCtx) {
        SPDLOG_WARN("申请重采样上下文失败");
        return false;
    }
    av_opt_set_int       (*swrCtx, "in_sample_rate",  frame->sample_rate,         0);
    av_opt_set_int       (*swrCtx, "out_sample_rate", 48000,                      0);
    av_opt_set_sample_fmt(*swrCtx, "in_sample_fmt",   decodeCodecCtx->sample_fmt, 0);
    av_opt_set_sample_fmt(*swrCtx, "out_sample_fmt",  AV_SAMPLE_FMT_S16,          0);
    #if FF_API_OLD_CHANNEL_LAYOUT
    av_opt_set_channel_layout(*swrCtx, "in_channel_layout",  frame->channel_layout, 0);
    av_opt_set_channel_layout(*swrCtx, "out_channel_layout", AV_CH_LAYOUT_MONO,     0);
    #else
    static AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
    av_opt_set_chlayout(*swrCtx, "in_channel_layout",  &frame->ch_layout, 0);
    av_opt_set_chlayout(*swrCtx, "out_channel_layout", &mono,             0);
    #endif
    if(swr_init(*swrCtx) != 0) {
        SPDLOG_WARN("打开重采样上下文失败");
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

static bool decode(AVFrame* frame, AVPacket* packet, SwrContext** swrCtx, AVCodecContext* decodeCodecCtx, std::vector<char>& buffer, std::ofstream& output) {
    if(avcodec_send_packet(decodeCodecCtx, packet) != 0) {
        return false;
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
    return true;
}

std::tuple<bool, std::string> lifuren::audio::toFile(const std::string& pcmFile) {
    AVFrame * frame { nullptr };
    AVPacket* packet{ nullptr };
    AVCodecContext* encodeCodecCtx{ nullptr };
    if(!open_encoder(&frame, &packet, &encodeCodecCtx)) {
        close_encoder(&frame, &packet, &encodeCodecCtx);
        return {false, {}};
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
        return {false, outputFile};
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
        SPDLOG_WARN("打开音频输入文件失败：{}", pcmFile);
        close_encoder(&frame, &packet, &encodeCodecCtx);
        close_output(&outputCtx);
        return {false, outputFile};
    }
    if(avformat_write_header(outputCtx, NULL) != 0) {
        SPDLOG_WARN("写入音频头部失败");
    }
    while(input.read(data.data(), buffer_size)) {
        size       = input.gcount();
        nb_samples = size / sample_size;
        frame->pts            = AV_NOPTS_VALUE;
        frame->format         = AV_SAMPLE_FMT_S16;
        #if FF_API_OLD_CHANNEL_LAYOUT
        frame->channel_layout = AV_CH_LAYOUT_MONO;
        #else
        frame->ch_layout      = AV_CHANNEL_LAYOUT_MONO;
        #endif
        frame->nb_samples     = nb_samples;
        frame->sample_rate    = 48000;
        if(av_frame_get_buffer(frame, 0) != 0) {
            av_frame_unref(frame);
            break;
        }
        std::memcpy(frame->buf[0]->data, data.data(), size);
        if(!encode(pts, nb_samples, frame, packet, encodeCodecCtx, outputCtx)) {
            av_frame_unref(frame);
            break;
        }
        av_frame_unref(frame);
    }
    encode(pts, nb_samples, NULL, packet, encodeCodecCtx, outputCtx);
    if(av_write_trailer(outputCtx)) {
        SPDLOG_WARN("写入音频尾部失败");
    }
    input.close();
    close_output(&outputCtx);
    close_encoder(&frame, &packet, &encodeCodecCtx);
    return {true, outputFile};
}

static bool open_encoder(AVFrame** frame, AVPacket** packet, AVCodecContext** encodeCodecCtx) {
    *frame = av_frame_alloc();
    if(!*frame) {
        SPDLOG_WARN("申请数据帧失败");
        return false;
    }
    *packet = av_packet_alloc();
    if(!*packet) {
        SPDLOG_WARN("申请数据包失败");
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
        SPDLOG_WARN("申请编码器上下文失败");
        return false;
    }
    #if defined(__MP3__) && (defined(__linux) || defined(__linux__))
    (*encodeCodecCtx)->sample_fmt     = AV_SAMPLE_FMT_S16P;
    #else
    (*encodeCodecCtx)->sample_fmt     = AV_SAMPLE_FMT_S16;
    #endif
    #if FF_API_OLD_CHANNEL_LAYOUT
    (*encodeCodecCtx)->channel_layout = AV_CH_LAYOUT_MONO;
    #else
    (*encodeCodecCtx)->ch_layout      = AV_CHANNEL_LAYOUT_MONO;
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
        SPDLOG_WARN("申请格式上下文失败：{}", file);
        return false;
    }
    if(avformat_alloc_output_context2(outputCtx, NULL, NULL, file.c_str()) < 0) {
        SPDLOG_WARN("打开格式上下文失败：{}", file);
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
        SPDLOG_WARN("打开音频轨道失败：{}", file);
        return false;
    }
    if(avcodec_parameters_from_context(stream->codecpar, encodeCodecCtx) < 0) {
        SPDLOG_WARN("设置音频配置失败：{}", file);
        return false;
    }
    if(avio_open(&(*outputCtx)->pb, file.c_str(), AVIO_FLAG_WRITE) < 0) {
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

static bool encode(int64_t& pts, const int64_t& nb_samples, AVFrame* frame, AVPacket* packet, AVCodecContext* encodeCodecCtx, AVFormatContext* outputCtx) {
    if(avcodec_send_frame(encodeCodecCtx, frame) != 0) {
        return false;
    }
    while(avcodec_receive_packet(encodeCodecCtx, packet) == 0) {
        packet->pts = pts;
        packet->dts = pts;
        pts += nb_samples;
        av_write_frame(outputCtx, packet);
        av_packet_unref(packet);
    }
    return true;
}

inline static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mag_pha_stft(
    torch::Tensor pcm_norm,
    int n_fft,
    int hop_size,
    int win_size,
    float compress_factor
) {
    auto window = torch::hann_window(win_size);
    auto spec = torch::stft(pcm_norm, n_fft, hop_size, win_size, window, true, "reflect", false, std::nullopt, true);
         spec = torch::view_as_real(spec);
    auto mag  = torch::sqrt(spec.pow(2).sum(-1) + (1e-9));
    auto pha  = torch::atan2(spec.index({"...", 1}), spec.index({"...", 0}) + (1e-5));
         mag  = torch::pow(mag, compress_factor);
    auto com  = torch::stack((mag * torch::cos(pha), mag * torch::sin(pha)), -1);
    return std::make_tuple<>(mag, pha, com);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lifuren::audio::pcm_mag_pha_stft(
    std::vector<short>& pcm,
    int n_fft,
    int hop_size,
    int win_size,
    float compress_factor
) {
    auto pcm_tensor = torch::from_blob(pcm.data(), {1, static_cast<int>(pcm.size())}, torch::kShort).to(torch::kFloat32);
         pcm_tensor = pcm_tensor / NORMALIZATION;
    return mag_pha_stft(pcm_tensor, n_fft, hop_size, win_size, compress_factor);
}

inline static torch::Tensor mag_pha_istft(
    torch::Tensor mag,
    torch::Tensor pha,
    int n_fft,
    int hop_size,
    int win_size,
    float compress_factor
) {
    auto window = torch::hann_window(win_size);
         mag = torch::pow(mag, (1.0 / compress_factor));
    auto com = torch::complex(mag * torch::cos(pha), mag * torch::sin(pha));
    return torch::istft(com, n_fft, hop_size, win_size, window, true);
}

std::vector<short> lifuren::audio::pcm_mag_pha_istft(
    torch::Tensor mag,
    torch::Tensor pha,
    int n_fft,
    int hop_size,
    int win_size,
    float compress_factor
) {
    auto result = mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor);
         result = result * NORMALIZATION;
    float* data = reinterpret_cast<float*>(result.data_ptr());
    std::vector<short> pcm;
    pcm.resize(result.sizes()[1]);
    std::copy_n(data, pcm.size(), pcm.data());
    return pcm;
}

bool lifuren::audio::embedding(const std::string& path, const std::string& dataset, std::ofstream& stream, lifuren::thread::ThreadPool& pool) {
    const std::string source = "source";
    const std::string target = "target";
    std::vector<std::string> files;
    lifuren::file::listFile(files, dataset, { ".aac", ".mp3", ".flac" });
    if(files.empty()) {
        SPDLOG_DEBUG("音频嵌入文件为空：{}", dataset);
        return true;
    }
    for(const auto& source_file : files) {
        const auto index = source_file.find_last_of('.');
        if(index == std::string::npos) {
            continue;
        }
        if(index < source.size()) {
            continue;
        }
        const auto label = source_file.substr(index - source.size(), source.size());
        if(label != source) {
            continue;
        }
        auto target_file(source_file);
        target_file.replace(index - source.size(), source.size(), target);
        const auto iterator = std::find(files.begin(), files.end(), target_file);
        if(iterator == files.end()) {
            SPDLOG_WARN("音频文件没有目标文件：{}", source_file);
            continue;
        }
        pool.submit([&stream, source_file, target_file]() {
            const auto [source_success, source_pcm] = lifuren::audio::toPcm(source_file);
            const auto [target_success, target_pcm] = lifuren::audio::toPcm(target_file);
            if(source_success && target_success) {
                SPDLOG_DEBUG("转换PCM成功：{} - {}", source_file, target_file);
                ::embedding(source_pcm, target_pcm, stream);
                std::filesystem::remove(source_pcm);
                std::filesystem::remove(target_pcm);
            } else {
                SPDLOG_DEBUG("转换PCM失败：{} - {}", source_file, target_file);
            }
        });
    }
    return true;
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
    SPDLOG_DEBUG("开始音频嵌入：{} - {}", source, target);
    std::vector<short> source_pcm;
    std::vector<short> target_pcm;
    source_pcm.resize(DATASET_PCM_LENGTH);
    target_pcm.resize(DATASET_PCM_LENGTH);
    while(
        source_stream.read(reinterpret_cast<char*>(source_pcm.data()), DATASET_PCM_LENGTH * sizeof(short)) &&
        target_stream.read(reinterpret_cast<char*>(target_pcm.data()), DATASET_PCM_LENGTH * sizeof(short)) &&
        source_stream.gcount() == target_stream.gcount()
    ) {
        // 短时傅里叶变换
        auto source_tuple = lifuren::audio::pcm_mag_pha_stft(source_pcm);
        auto target_tuple = lifuren::audio::pcm_mag_pha_stft(target_pcm);
        {
            // 保证每次都是成对写入
            std::lock_guard<std::mutex> lock(embedding_mutex);
            ::embedding(stream, source_tuple);
            ::embedding(stream, target_tuple);
        }
    }
    source_stream.close();
    target_stream.close();
    stream.flush();
    SPDLOG_DEBUG("音频嵌入完成：{} - {}", source, target);
}

inline static void embedding(std::ofstream& stream, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& tuple) {
    auto tensor = torch::stack({
        std::get<0>(tuple),
        std::get<1>(tuple)
    }).squeeze();
    stream.write(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel() * tensor.element_size());
}

lifuren::dataset::FileDatasetLoader lifuren::audio::loadFileDatasetLoader(const size_t batch_size, const std::string& path) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        [](const std::string& file, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) {
            std::ifstream stream;
            stream.open(file, std::ios_base::binary);
            if(!stream.is_open()) {
                SPDLOG_WARN("音频嵌入文件打开失败：{}", file);
                stream.close();
                return;
            }
            // [mag, pha] = [1, 201, 5] + [1, 201, 5] = [2, 201, 5]
            torch::Tensor source_tensor = torch::zeros({ 2, 201, 5 }, torch::kFloat32).to(device);
            torch::Tensor target_tensor = torch::zeros({ 2, 201, 5 }, torch::kFloat32).to(device);
            const auto size = source_tensor.numel() * source_tensor.element_size();
            while(
                stream.read(reinterpret_cast<char*>(source_tensor.data_ptr()), size) &&
                stream.read(reinterpret_cast<char*>(target_tensor.data_ptr()), size)
            ) {
                features.push_back(source_tensor.clone());
                labels  .push_back(target_tensor.clone());
            }
            stream.close();
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

bool lifuren::audio::datasetPreprocessing(const std::string& path) {
    return lifuren::dataset::allDatasetPreprocessing(path, lifuren::config::EMBEDDING_MODEL_FILE, &lifuren::audio::embedding);
}
