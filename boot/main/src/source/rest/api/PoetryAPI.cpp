#include "lifuren/REST.hpp"

#include "httplib.h"

// 生成诗词
static void restPostPoetry();

void lifuren::restPoetryAPI() {
    restPostPoetry();
}

static void restPostPoetry() {
    lifuren::httpServer.Post("/poetry/generate", [](const httplib::Request& request, httplib::Response& response) {
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
