#include "lifuren/REST.hpp"

#include "lifuren/Ptr.hpp"
#include "lifuren/Client.hpp"

#include "spdlog/spdlog.h"

static lifuren::ChatClient* clientPtr{ nullptr };

static void initChatClient();
static void restPostChat();
static void restPostChatStream();

void lifuren::restChatAPI() {
    initChatClient();
    restPostChat();
    restPostChatStream();
}

static void initChatClient() {
    // TODO: 刷新配置重新修改
    LFR_DELETE_PTR(clientPtr);
    lifuren::options::RestChatOptions options;
    const auto& config = lifuren::config::CONFIG;
    const auto& chat   = config.chat;
    if(chat.client == lifuren::config::CONFIG_OLLAMA) {
        options.of(config.ollama);
        clientPtr = new lifuren::OllamaChatClient{options};
    } else {
    }
}

static void restPostChat() {
    lifuren::httpServer.Post("/chat", [](const httplib::Request& request, httplib::Response& response) {
        if(clientPtr == nullptr) {
            lifuren::response(response, "9999", "没有聊天终端");
            return;
        }
        nlohmann::json body = nlohmann::json::parse(request.body);
        if(body.find("prompt") == body.end()) {
            lifuren::response(response, "9999", "缺少聊天内容");
            return;
        }
        std::string prompt  = body["prompt"];
        std::string message = clientPtr->chat(prompt);
        lifuren::response(response, message.c_str());
    });
}

static void restPostChatStream() {
    lifuren::httpServer.Post("/chat/stream", [](const httplib::Request& request, httplib::Response& response) {
        if(clientPtr == nullptr) {
            lifuren::response(response, "9999", "没有聊天终端");
            return;
        }
        nlohmann::json body = nlohmann::json::parse(request.body);
        if(body.find("prompt") == body.end()) {
            lifuren::response(response, "9999", "缺少聊天内容");
            return;
        }
        std::string prompt = body["prompt"];
        response.set_chunked_content_provider(lifuren::content::type::EVENT, [prompt](size_t, httplib::DataSink& sink) {
            clientPtr->chat(prompt, [&sink](const char* data, size_t length, bool done) {
                nlohmann::json body = lifuren::buildResponse(data);
                body["done"] = done;
                std::string message = "data:" + body.dump() + "\n\n";
                sink.write(message.c_str(), message.size());
                if(done) {
                    sink.done();
                }
                return true;
            });
            return true;
        });
    });
}
