#include "lifuren/REST.hpp"

#include "httplib.h"

#include "nlohmann/json.hpp"

// 生成视频
static void restPostVideoGenerate();

void lifuren::restVideoAPI() {
    restPostVideoGenerate();
}

static void restPostVideoGenerate() {
    lifuren::httpServer.Post("/video/generate", [](const httplib::Request& request, httplib::Response& response, const httplib::ContentReader& content_reader) {
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
            return file.name == "video";
        });
        if(iterator == files.end()) {
            lifuren::response(response, "4415", "缺少视频文件");
            return;
        }
        httplib::MultipartFormData video = *iterator;
        if(!(
            video.content_type == "video/mp4"
        )) {
            lifuren::response(response, "4415", "视频格式错误");
            return;
        }
        response.set_content_provider(video.content_type, [video = std::move(video)](size_t /* offset */, httplib::DataSink& sink) {
            sink.write(video.content.data(), video.content.size());
            sink.done();
            return true;
        });
    });
}
