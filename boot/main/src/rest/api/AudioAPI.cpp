#include "lifuren/REST.hpp"

#include "httplib.h"

// 生成音频
static void restPostAudioGenerate();

void lifuren::restAudioAPI() {
    restPostAudioGenerate();
}

static void restPostAudioGenerate() {
    lifuren::httpServer.Post("/audio/generate", [](const httplib::Request& request, httplib::Response& response, const httplib::ContentReader& content_reader) {
        httplib::MultipartFormDataItems files;
        content_reader(
            [&](const httplib::MultipartFormData& file) {
                files.push_back(file);
                return true;
            },
            [&](const char* data, size_t data_length) {
                files.back().content.append(data, data_length);
                return true;
            }
        );
        auto iterator = std::find_if(files.begin(), files.end(), [](const auto& file) {
            return file.name == "audio";
        });
        if(iterator == files.end()) {
            lifuren::response(response, "4415", "缺少音频文件");
            return;
        }
        httplib::MultipartFormData audio = *iterator;
        if(!(
            audio.content_type == "audio/aac" ||
            audio.content_type == "audio/ogg" ||
            audio.content_type == "audio/wav" ||
            audio.content_type == "audio/mpeg" // MP3
        )) {
            lifuren::response(response, "4415", "音频格式错误");
            return;
        }
        response.set_content_provider(audio.content_type, [audio = std::move(audio)](size_t /* offset */, httplib::DataSink& sink) {
            sink.write(audio.content.data(), audio.content.size());
            sink.done();
            return true;
        });
    });
}
