#include "lifuren/Dataset.hpp"

#include <chrono>
#include <fstream>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Audio.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"

extern "C" {

#include "libavutil/opt.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswresample/swresample.h"

}

#ifndef LFR_SAMPLE_RATE
#define LFR_SAMPLE_RATE 48000
#endif

#ifndef LFR_MONO_NB_CHANNELS
#define LFR_MONO_NB_CHANNELS 1
#endif

#ifndef LFR_OLD_FFMPEG
#define LFR_OLD_FFMPEG LIBAVUTIL_VERSION_MAJOR < 58
#endif

// 解码
static bool decode(AVFrame* frame, AVPacket* packet, SwrContext** swrCtx, AVCodecContext* decodeCodecCtx, std::vector<char>& buffer, std::ofstream& output);
static bool open_swr (SwrContext** swrCtx, AVFrame* frame);
static void close_swr(SwrContext** swrCtx);
static bool open_input (AVPacket** packet, AVFormatContext** inputCtx, const std::string& file);
static void close_input(AVPacket** packet, AVFormatContext** inputCtx);
static bool open_decoder (AVFrame** frame, AVCodecContext** decodeCodecCtx, const std::string& suffix);
static void close_decoder(AVFrame** frame, AVCodecContext** decodeCodecCtx);

// 编码
static bool encode(int64_t& pts, const int64_t nb_samples, AVFrame* frame, AVPacket* packet, AVCodecContext* encodeCodecCtx, AVFormatContext* outputCtx);
static bool open_output (AVFormatContext** outputCtx, AVCodecContext* encodeCodecCtx, int& stream_index, const std::string& file);
static void close_output(AVFormatContext** outputCtx);
static bool open_encoder (AVFrame** frame, AVPacket** packet, AVCodecContext** encodeCodecCtx);
static void close_encoder(AVFrame** frame, AVPacket** packet, AVCodecContext** encodeCodecCtx);

// 嵌入
static void embedding_shikuang(std::ofstream& stream, const std::string& source);

const static float NORMALIZATION = 32768.0F;

std::tuple<bool, std::string> lifuren::dataset::audio::toPcm(const std::string& audioFile) {
    AVPacket       * packet  { nullptr };
    AVFormatContext* inputCtx{ nullptr };
    if(!open_input(&packet, &inputCtx, audioFile)) {
        close_input(&packet, &inputCtx);
        return { false, {} };
    }
    const auto pos = audioFile.find_last_of('.');
    if(pos == std::string::npos) {
        SPDLOG_WARN("音频输入文件格式错误：{}", audioFile);
        return { false, {} };
    }
    const auto suffix  = audioFile.substr(   pos + 1);
    const auto pcmFile = audioFile.substr(0, pos + 1) + "pcm";
    AVFrame       * frame         { nullptr };
    AVCodecContext* decodeCodecCtx{ nullptr };
    if(!open_decoder(&frame, &decodeCodecCtx, suffix)) {
        close_input(&packet, &inputCtx);
        close_decoder(&frame, &decodeCodecCtx);
        return { false, pcmFile };
    }
    std::ofstream output;
    output.open(pcmFile, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if(!output.is_open()) {
        SPDLOG_WARN("音频输出文件打开失败：{}", pcmFile);
        close_input(&packet, &inputCtx);
        close_decoder(&frame, &decodeCodecCtx);
        return { false, pcmFile };
    }
    std::vector<char> buffer;
    buffer.resize(16 * 1024);
    SwrContext* swrCtx{ nullptr };
    bool success = true;
    while(av_read_frame(inputCtx, packet) == 0) {
        if(!decode(frame, packet, &swrCtx, decodeCodecCtx, buffer, output)) {
            success = false;
            av_packet_unref(packet);
            break;
        }
        av_packet_unref(packet);
    }
    if(success) {
        decode(frame, NULL, &swrCtx, decodeCodecCtx, buffer, output);
    }
    output.close();
    close_swr(&swrCtx);
    close_input(&packet, &inputCtx);
    close_decoder(&frame, &decodeCodecCtx);
    return { success, pcmFile };
}

static bool open_input(AVPacket** packet, AVFormatContext** inputCtx, const std::string& file) {
    *packet = av_packet_alloc();
    if(!*packet) {
        SPDLOG_WARN("数据包申请失败：{}", file);
        return false;
    }
    *inputCtx = avformat_alloc_context();
    if(!*inputCtx) {
        SPDLOG_WARN("格式上下文申请失败：{}", file);
        return false;
    }
    if(avformat_open_input(inputCtx, file.c_str(), NULL, NULL) != 0) {
        SPDLOG_WARN("格式上下文打开失败：{}", file);
        return false;
    }
    if((*inputCtx)->nb_streams != 1) {
        SPDLOG_WARN("音频文件包含多个轨道：{}", file);
        return false;
    }
    if((*inputCtx)->streams[0]->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) {
        SPDLOG_WARN("音频文件没有音频轨道：{}", file);
        return false;
    }
    return true;
}

static void close_input(AVPacket** packet, AVFormatContext** inputCtx) {
    if(*packet) {
        av_packet_free(packet);
    }
    if(*inputCtx) {
        // avformat_free_context(*inputCtx);
        avformat_close_input(inputCtx);
    }
}

static bool open_decoder(AVFrame** frame, AVCodecContext** decodeCodecCtx, const std::string& suffix) {
    *frame = av_frame_alloc();
    if(!*frame) {
        SPDLOG_WARN("数据帧申请失败：{}", suffix);
        return false;
    }
    const AVCodec* decoder { nullptr };
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
        SPDLOG_WARN("解码器上下文申请失败：{}", suffix);
        return false;
    }
    if(avcodec_open2(*decodeCodecCtx, decoder, nullptr) != 0) {
        SPDLOG_WARN("解码器上下文打开失败：{}", suffix);
        return false;
    }
    return true;
}

static void close_decoder(AVFrame** frame, AVCodecContext** decodeCodecCtx) {
    if(*frame) {
        av_frame_free(frame);
    }
    if(*decodeCodecCtx) {
        // avcodec_close(*decodeCodecCtx);
        avcodec_free_context(decodeCodecCtx);
    }
}

static bool open_swr(SwrContext** swrCtx, AVFrame* frame) {
    const AVSampleFormat format = static_cast<AVSampleFormat>(frame->format);
    #if LFR_OLD_FFMPEG
    auto mono = AV_CH_LAYOUT_MONO;
    SPDLOG_DEBUG(
        "重采样信息：{} - {} - {} -> {} - {} - {}",
        frame->channel_layout, av_get_sample_fmt_name(format),            frame->sample_rate,
        mono,                  av_get_sample_fmt_name(AV_SAMPLE_FMT_S16), LFR_SAMPLE_RATE
    );
    swr_alloc_set_opts(
        *swrCtx,
        mono,                  AV_SAMPLE_FMT_S16, LFR_SAMPLE_RATE,
        frame->channel_layout, format,            frame->sample_rate,
        0, nullptr
    );
    #else
    static AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
    SPDLOG_DEBUG(
        "重采样信息：{} - {} - {} -> {} - {} - {}",
        frame->ch_layout.nb_channels, av_get_sample_fmt_name(format),            frame->sample_rate,
        mono.nb_channels,             av_get_sample_fmt_name(AV_SAMPLE_FMT_S16), LFR_SAMPLE_RATE
    );
    swr_alloc_set_opts2(
        swrCtx,
        &mono,             AV_SAMPLE_FMT_S16, LFR_SAMPLE_RATE,
        &frame->ch_layout, format,            frame->sample_rate,
        0, nullptr
    );
    #endif
    if(!*swrCtx) {
        SPDLOG_WARN("重采样上下文申请失败");
        return false;
    }
    if(swr_init(*swrCtx) != 0) {
        SPDLOG_WARN("重采样上下文打开失败");
        return false;
    }
    return true;
}

static void close_swr(SwrContext** swrCtx) {
    if(*swrCtx) {
        // swr_close(*swrCtx);
        swr_free(swrCtx);
    }
}

static bool decode(AVFrame* frame, AVPacket* packet, SwrContext** swrCtx, AVCodecContext* decodeCodecCtx, std::vector<char>& buffer, std::ofstream& output) {
    if(avcodec_send_packet(decodeCodecCtx, packet) != 0) {
        return false;
    }
    while(avcodec_receive_frame(decodeCodecCtx, frame) == 0) {
        if(!*swrCtx && !open_swr(swrCtx, frame)) {
            av_frame_unref(frame);
            return false;
        }
        uint8_t* buffer_data = reinterpret_cast<uint8_t*>(buffer.data());
        const int swr_size = swr_convert(
            *swrCtx,
            &buffer_data,
            swr_get_out_samples(*swrCtx, frame->nb_samples),
            const_cast<const uint8_t**>(frame->data),
            frame->nb_samples
        );
        av_frame_unref(frame);
        const int buffer_size = av_samples_get_buffer_size(NULL, LFR_MONO_NB_CHANNELS, swr_size, AV_SAMPLE_FMT_S16, 0);
        output.write(buffer.data(), buffer_size);
    }
    return true;
}

std::tuple<bool, std::string> lifuren::dataset::audio::toFile(const std::string& pcmFile) {
    AVFrame * frame { nullptr };
    AVPacket* packet{ nullptr };
    AVCodecContext* encodeCodecCtx{ nullptr };
    if(!open_encoder(&frame, &packet, &encodeCodecCtx)) {
        close_encoder(&frame, &packet, &encodeCodecCtx);
        return { false, {} };
    }
    int stream_index;
    AVFormatContext* outputCtx{ nullptr };
    const auto pos = pcmFile.find_last_of('.');
    if(pos == std::string::npos) {
        SPDLOG_WARN("音频输入文件格式错误：{}", pcmFile);
        return { false, {} };
    }
    #ifdef __MP3__
    auto outputFile = pcmFile.substr(0, pos + 1) + "mp3";
    #else
    auto outputFile = pcmFile.substr(0, pos + 1) + "flac";
    #endif
    if(!open_output(&outputCtx, encodeCodecCtx, stream_index, outputFile)) {
        close_encoder(&frame, &packet, &encodeCodecCtx);
        close_output(&outputCtx);
        return { false, outputFile };
    }
    const int sample_size = av_get_bytes_per_sample(AV_SAMPLE_FMT_S16);
    #ifdef __MP3__
    const int buffer_size = av_samples_get_buffer_size(NULL, LFR_MONO_NB_CHANNELS, 1152, AV_SAMPLE_FMT_S16, 0);
    #else
    const int buffer_size = av_samples_get_buffer_size(NULL, LFR_MONO_NB_CHANNELS, 4608, AV_SAMPLE_FMT_S16, 0);
    #endif
    int64_t pts = 0;
    int64_t size;
    int64_t nb_samples;
    std::vector<char> data;
    data.resize(buffer_size);
    std::ifstream input;
    input.open(pcmFile, std::ios_base::in | std::ios_base::binary);
    if(!input.is_open()) {
        SPDLOG_WARN("音频输入文件打开失败：{}", pcmFile);
        close_encoder(&frame, &packet, &encodeCodecCtx);
        close_output(&outputCtx);
        return { false, outputFile };
    }
    if(avformat_write_header(outputCtx, NULL) != 0) {
        SPDLOG_WARN("音频文件头部写入失败：{}", outputFile);
    }
    bool success = true;
    while(input.read(data.data(), buffer_size)) {
        size       = input.gcount();
        nb_samples = size / sample_size;
        frame->pts            = AV_NOPTS_VALUE;
        frame->format         = AV_SAMPLE_FMT_S16;
        #if LFR_OLD_FFMPEG
        frame->channel_layout = AV_CH_LAYOUT_MONO;
        #else
        frame->ch_layout      = AV_CHANNEL_LAYOUT_MONO;
        #endif
        frame->nb_samples     = nb_samples;
        frame->sample_rate    = LFR_SAMPLE_RATE;
        if(av_frame_get_buffer(frame, 0) != 0) {
            success = false;
            av_frame_unref(frame);
            break;
        }
        std::memcpy(frame->buf[0]->data, data.data(), size);
        if(!encode(pts, nb_samples, frame, packet, encodeCodecCtx, outputCtx)) {
            success = false;
            av_frame_unref(frame);
            break;
        }
        av_frame_unref(frame);
    }
    if(success) {
        encode(pts, nb_samples, NULL, packet, encodeCodecCtx, outputCtx);
    }
    if(av_write_trailer(outputCtx)) {
        SPDLOG_WARN("音频文件尾部写入失败：{}", outputFile);
    }
    input.close();
    close_output(&outputCtx);
    close_encoder(&frame, &packet, &encodeCodecCtx);
    return { success, outputFile };
}

static bool open_encoder(AVFrame** frame, AVPacket** packet, AVCodecContext** encodeCodecCtx) {
    *frame = av_frame_alloc();
    if(!*frame) {
        SPDLOG_WARN("数据帧申请失败");
        return false;
    }
    *packet = av_packet_alloc();
    if(!*packet) {
        SPDLOG_WARN("数据包申请失败");
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
        SPDLOG_WARN("编码器上下文申请失败");
        return false;
    }
    #if defined(__MP3__) && (defined(__linux) || defined(__linux__))
    (*encodeCodecCtx)->sample_fmt     = AV_SAMPLE_FMT_S16P;
    #else
    (*encodeCodecCtx)->sample_fmt     = AV_SAMPLE_FMT_S16;
    #endif
    #if LFR_OLD_FFMPEG
    (*encodeCodecCtx)->channel_layout = AV_CH_LAYOUT_MONO;
    #else
    (*encodeCodecCtx)->ch_layout      = AV_CHANNEL_LAYOUT_MONO;
    #endif
    (*encodeCodecCtx)->sample_rate    = LFR_SAMPLE_RATE;
    if(avcodec_open2(*encodeCodecCtx, encoder, nullptr) != 0) {
        SPDLOG_WARN("编码器上下文打开失败");
        return false;
    }
    return true;
}

static void close_encoder(AVFrame** frame, AVPacket** packet, AVCodecContext** encodeCodecCtx) {
    if(*frame) {
        av_frame_free(frame);
    }
    if(*packet) {
        av_packet_free(packet);
    }
    if(*encodeCodecCtx) {
        // avcodec_close(*encodeCodecCtx);
        avcodec_free_context(encodeCodecCtx);
    }
}

static bool open_output(AVFormatContext** outputCtx, AVCodecContext* encodeCodecCtx, int& stream_index, const std::string& file) {
    *outputCtx = avformat_alloc_context();
    if(!*outputCtx) {
        SPDLOG_WARN("格式上下文申请失败：{}", file);
        return false;
    }
    if(avformat_alloc_output_context2(outputCtx, NULL, NULL, file.c_str()) < 0) {
        SPDLOG_WARN("格式上下文打开失败：{}", file);
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
        SPDLOG_WARN("音频编码参数配置失败：{}", file);
        return false;
    }
    if(avio_open(&(*outputCtx)->pb, file.c_str(), AVIO_FLAG_WRITE) < 0) {
        SPDLOG_WARN("音频输出文件打开失败：{}", file);
        return false;
    }
    stream_index = stream->index;
    return true;
}

static void close_output(AVFormatContext** outputCtx) {
    if(*outputCtx) {
        // avformat_free_context(*outputCtx);
        avformat_close_input(outputCtx);
    }
}

static bool encode(int64_t& pts, const int64_t nb_samples, AVFrame* frame, AVPacket* packet, AVCodecContext* encodeCodecCtx, AVFormatContext* outputCtx) {
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

torch::Tensor lifuren::dataset::audio::pcm_stft(
    std::vector<short>& pcm,
    int n_fft,
    int hop_size,
    int win_size
) {
    auto tensor = torch::from_blob(pcm.data(), { 1, static_cast<int>(pcm.size()) }, torch::kShort).to(torch::kFloat32);
         tensor = tensor / NORMALIZATION;
    // auto norm_factor = torch::sqrt(tensor.sizes()[1] / torch::sum(tensor.pow(2.0)));
    //      tensor = tensor * norm_factor;
    auto wind = torch::hann_window(win_size);
    auto real = torch::view_as_real(torch::stft(tensor, n_fft, hop_size, win_size, wind, true, "reflect", false, std::nullopt, true));
    // // 幅度
    auto mag = torch::sqrt(real.pow(2).sum(-1));
    // // 相位
    auto pha = torch::atan2(real.index({"...", 1}), real.index({"...", 0}));
    // auto com = torch::stack({mag, pha}, -1);
    auto com = torch::stack({mag * torch::cos(pha), mag * torch::sin(pha)}, -1);
    return com;
}

std::vector<short> lifuren::dataset::audio::pcm_istft(
    const torch::Tensor& tensor,
    int n_fft,
    int hop_size,
    int win_size
) {
    auto mag = tensor.index({"...", 0});
    auto pha = tensor.index({"...", 1});
    auto com = torch::complex(mag, pha);
    // auto com = torch::complex(mag * torch::cos(pha), mag * torch::sin(pha));
    auto wind   = torch::hann_window(win_size);
    auto result = torch::istft(com, n_fft, hop_size, win_size, wind, true);
    // auto result = torch::istft(torch::view_as_complex(tensor), n_fft, hop_size, win_size, wind, true);
         result = result * NORMALIZATION;
    float* data = reinterpret_cast<float*>(result.data_ptr());
    std::vector<short> pcm;
    pcm.resize(result.sizes()[1]);
    std::copy_n(data, pcm.size(), pcm.data());
    return pcm;
}

bool lifuren::dataset::audio::embedding_shikuang(const std::string& path, const std::string& dataset, std::ofstream& stream, lifuren::thread::ThreadPool& pool) {
    std::vector<std::string> files;
    lifuren::file::list_file(files, dataset, { ".aac", ".mp3", ".pcm", ".flac" });
    if(files.empty()) {
        SPDLOG_DEBUG("音频嵌入文件为空：{}", dataset);
        return true;
    }
    for(const auto& source_file : files) {
        pool.submit([&stream, source_file]() {
            const auto source_suffix = lifuren::file::file_suffix(source_file);
            if(source_suffix == ".pcm") {
                ::embedding_shikuang(stream, source_file);
            } else {
                const auto [source_success, source_pcm] = lifuren::dataset::audio::toPcm(source_file);
                if(source_success) {
                    SPDLOG_DEBUG("转换PCM成功：{}", source_file);
                    ::embedding_shikuang(stream, source_pcm);
                    std::filesystem::remove(source_pcm);
                } else {
                    SPDLOG_DEBUG("转换PCM失败：{}", source_file);
                }
            }
        });
    }
    return true;
}

static void embedding_shikuang(std::ofstream& stream, const std::string& source) {
    std::ifstream source_stream;
    source_stream.open(source, std::ios_base::binary);
    if(!source_stream.is_open()) {
        SPDLOG_DEBUG("音频文件打开失败：{}", source);
        source_stream.close();
        return;
    }
    SPDLOG_DEBUG("开始嵌入音频文件：{}", source);
    std::vector<short> source_pcm;
    source_pcm.resize(LFR_AUDIO_PCM_LENGTH);
    while(source_stream.read(reinterpret_cast<char*>(source_pcm.data()), LFR_AUDIO_PCM_LENGTH * sizeof(short))) {
        auto source_tensor = lifuren::dataset::audio::pcm_stft(source_pcm);
        {
            static std::mutex embedding_mutex;
            std::lock_guard<std::mutex> lock(embedding_mutex);
            lifuren::write_tensor(stream, source_tensor);
        }
    }
    source_stream.close();
    stream.flush();
    SPDLOG_DEBUG("音频文件嵌入完成：{}", source);
}

lifuren::dataset::SeqDatasetLoader lifuren::dataset::audio::loadShikuangDatasetLoader(const size_t batch_size, const std::string& path) {
    auto dataset = lifuren::dataset::Dataset(
        lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
        { ".embedding" },
        [](
            const std::string         & file,
            std::vector<torch::Tensor>& labels,
            std::vector<torch::Tensor>& features,
            const torch::DeviceType   & device
        ) {
            std::ifstream stream;
            stream.open(file, std::ios_base::binary);
            if(!stream.is_open()) {
                SPDLOG_WARN("音频嵌入文件打开失败：{}", file);
                stream.close();
                return;
            }
            while(true) {
                auto tensor = lifuren::read_tensor(stream);
                if(stream.eof()) {
                    break;
                }
                labels.push_back(tensor.clone().to(device));
                features.push_back(tensor.clone().to(device));
            }
            stream.close();
        }
    ).map(torch::data::transforms::Stack<>());
    torch::data::DataLoaderOptions options(batch_size);
    options.drop_last() = true;
    return torch::data::make_data_loader<LFT_SEQ_SAMPLER>(std::move(dataset), std::move(options));
}

bool lifuren::audio::allDatasetPreprocessShikuang(const std::string& path) {
    return lifuren::dataset::allDatasetPreprocess(path, lifuren::config::LIFUREN_EMBEDDING_FILE, &lifuren::dataset::audio::embedding_shikuang);
}
