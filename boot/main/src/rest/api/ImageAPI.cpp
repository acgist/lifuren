#include "lifuren/REST.hpp"

#include "httplib.h"

#include "nlohmann/json.hpp"

// 生成图片
static void restPostImageGenerate();

void lifuren::restImageAPI() {
    restPostImageGenerate();
}

static void restPostImageGenerate() {
    lifuren::httpServer.Post("/image/generate", [](const httplib::Request& request, httplib::Response& response, const httplib::ContentReader& content_reader) {
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
            return file.name == "image";
        });
        if(iterator == files.end()) {
            lifuren::response(response, "4415", "缺少图片文件");
            return;
        }
        httplib::MultipartFormData image = *iterator;
        if(!(
            image.content_type == "image/png" ||
            image.content_type == "image/jpeg"
        )) {
            lifuren::response(response, "4415", "图片格式错误");
            return;
        }
        response.set_content_provider(image.content_type, [image = std::move(image)](size_t /* offset */, httplib::DataSink& sink) {
            sink.write(image.content.data(), image.content.size());
            sink.done();
            return true;
        });
    });
}
