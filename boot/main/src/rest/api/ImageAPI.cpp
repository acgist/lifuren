#include "lifuren/REST.hpp"

#include "httplib.h"

#include "nlohmann/json.hpp"

// 生成图片
static void restPostImage();

void lifuren::restImageAPI() {
    restPostImage();
}

static void restPostImage() {
    lifuren::httpServer.Post("/image/generate", [](const httplib::Request& request, httplib::Response& response) {
        if(request.body.empty()) {
            return;
        }
        nlohmann::json body = nlohmann::json::parse(request.body);
        const auto& prompt = body.find("prompt");
        if(prompt == body.end()) {
            lifuren::response(response, "9999", "缺少提示内容");
            return;
        }
        // 图片内容
        if(request.has_file("image")) {
        }
        lifuren::response(response, "");
    });
}
