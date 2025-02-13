#include "lifuren/REST.hpp"

#include <fstream>

#include "httplib.h"

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/audio/Audio.hpp"
#include "lifuren/video/Video.hpp"

static void restPostAudioGenerate();
static void restPostVideoGenerate();

static void recvFile(httplib::MultipartFormDataItems& files, const httplib::ContentReader& content_reader);
static bool sendFile(const std::string& file, httplib::DataSink& sink);

void lifuren::restModelAPI() {
    restPostAudioGenerate();
    restPostVideoGenerate();
}

static void restPostAudioGenerate() {
    lifuren::restServer.Post("/audio/generate", [](const httplib::Request& request, httplib::Response& response, const httplib::ContentReader& content_reader) {
        httplib::MultipartFormDataItems files;
        recvFile(files, content_reader);
        auto iterator = std::find_if(files.begin(), files.end(), [](const auto& file) {
            return file.name == "audio";
        });
        if(iterator == files.end()) {
            lifuren::response(response, "1400", "缺少音频文件");
            return;
        }
        httplib::MultipartFormData& audio = *iterator;
        if(!(
            audio.content_type == "audio/aac" ||
            audio.content_type == "audio/ogg" ||
            audio.content_type == "audio/wav" ||
            audio.content_type == "audio/mp3"
        )) {
            lifuren::response(response, "1415", "音频格式错误");
            return;
        }
        const std::string path = request.get_param_value("path");
        if(path.empty()) {
            lifuren::response(response, "1400", "缺少模型路径");
            return;
        }
        const std::string model = request.get_param_value("model");
        if(model.empty()) {
            lifuren::response(response, "1400", "缺少终端类型");
            return;
        }
        auto client = lifuren::audio::getAudioClient(model);
        if(!client) {
            lifuren::response(response, "2400", "不支持的终端类型");
            return;
        }
        if(!client->load(path)) {
            lifuren::response(response, "2400", "模型加载失败");
            return;
        }
        lifuren::audio::AudioParams params {
            .audio = lifuren::file::join({ lifuren::config::CONFIG.tmp, audio.filename }).string()
        };
        const auto [success, output_file] = client->pred(params);
        if(!success) {
            lifuren::response(response, "2500", "音频生成失败");
            return;
        }
        response.set_content_provider(audio.content_type, [audio = std::move(audio), output_file = std::move(output_file)](size_t /* offset */, httplib::DataSink& sink) {
            sendFile(output_file, sink);
            sink.done();
            return true;
        });
    });
}

static void restPostVideoGenerate() {
    lifuren::restServer.Post("/video/generate", [](const httplib::Request& request, httplib::Response& response, const httplib::ContentReader& content_reader) {
        httplib::MultipartFormDataItems files;
        recvFile(files, content_reader);
        auto iterator = std::find_if(files.begin(), files.end(), [](const auto& file) {
            return file.name == "video";
        });
        if(iterator == files.end()) {
            lifuren::response(response, "1400", "缺少视频文件");
            return;
        }
        httplib::MultipartFormData& video = *iterator;
        if(!(
            video.content_type == "video/mp4"
        )) {
            lifuren::response(response, "1415", "视频格式错误");
            return;
        }
        const std::string path = request.get_param_value("path");
        if(path.empty()) {
            lifuren::response(response, "1400", "缺少模型路径");
            return;
        }
        const std::string model = request.get_param_value("model");
        if(model.empty()) {
            lifuren::response(response, "1400", "缺少终端类型");
            return;
        }
        auto client = lifuren::video::getVideoClient(model);
        if(!client) {
            lifuren::response(response, "2400", "不支持的终端类型");
            return;
        }
        if(!client->load(path)) {
            lifuren::response(response, "2400", "模型加载失败");
            return;
        }
        lifuren::video::VideoParams params {
            .video = lifuren::file::join({ lifuren::config::CONFIG.tmp, video.filename }).string()
        };
        const auto [success, output_file] = client->pred(params);
        if(!success) {
            lifuren::response(response, "2500", "视频生成失败");
            return;
        }
        response.set_content_provider(video.content_type, [video = std::move(video), output_file = std::move(output_file)](size_t /* offset */, httplib::DataSink& sink) {
            sendFile(output_file, sink);
            sink.done();
            return true;
        });
    });
}

inline static void recvFile(httplib::MultipartFormDataItems& files, const httplib::ContentReader& content_reader) {
    std::ofstream stream;
    content_reader(
        [&files, &stream](const httplib::MultipartFormData& file) {
            if(stream.is_open()) {
                SPDLOG_DEBUG("关闭保存文件");
                stream.close();
            }
            stream.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, file.filename }), std::ios_base::trunc | std::ios_base::binary);
            if(stream.is_open()) {
                SPDLOG_DEBUG("打开保存文件：{}", file.filename);
                files.push_back(file);
                return true;
            } else {
                SPDLOG_WARN("保存文件失败：{}", file.filename);
                return false;
            }
        },
        [&stream](const char* data, size_t data_length) {
            if(stream.is_open()) {
                stream.write(data, data_length);
            }
            return true;
        }
    );
    if(stream.is_open()) {
        SPDLOG_DEBUG("关闭保存文件");
        stream.close();
    }
}

inline static bool sendFile(const std::string& file, httplib::DataSink& sink) {
    std::ifstream stream;
    stream.open(file, std::ios_base::binary);
    if(!stream.is_open()) {
        std::string message = "打开文件失败：" + file;
        sink.write(message.data(), message.size());
        stream.close();
        return false;
    }
    constexpr static int SIZE = 8 * 1024;
    std::vector<char> data;
    data.resize(SIZE);
    while(stream.read(data.data(), SIZE)) {
        sink.write(data.data(), stream.gcount());
    }
    stream.close();
    return true;
}
