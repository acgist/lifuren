#include "lifuren/REST.hpp"

#include "httplib.h"

#include "nlohmann/json.hpp"

// 生成诗词
static void restPostPoetryGenerate();

void lifuren::restPoetryAPI() {
    restPostPoetryGenerate();
}

static void restPostPoetryGenerate() {
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
        lifuren::response(response, "");
    });
}
