#include "lifuren/Client.hpp"

#include <format>

#include "nlohmann/json.hpp"

static std::string body(const std::string& prompt, bool stream, const lifuren::RestChatOptions& options);

lifuren::RestChatClient::RestChatClient(lifuren::RestChatOptions options) : options(options) {
    this->restClient = std::make_unique<lifuren::RestClient>(options.api);
}

lifuren::RestChatClient::~RestChatClient() {
}

std::string lifuren::RestChatClient::chat(const std::string& prompt) {
    auto response = this->restClient->postJson(this->options.path, body(prompt, false, this->options));
    // TODO: list
    auto json = nlohmann::json::parse(response->body);
    if(json.find("error") != json.end()) {
        return json["error"].template get<std::string>();
    }
    if(json.find("message") != json.end()) {
        auto& message = json["message"];
        return message["content"];
    }
    return "没有响应";
}

void lifuren::RestChatClient::chat(const std::string& prompt, std::function<bool(const char*, size_t, bool)> callback) {
    this->restClient->postStream(this->options.path, body(prompt, true, this->options), [&callback](const char* text, size_t length) {
        auto json = nlohmann::json::parse(std::string(text, length));
        bool done = true;
        std::string response;
        if(json.find("error") != json.end()) {
            response = json["error"].template get<std::string>();
        } else if(json.find("message") != json.end()) {
            auto& message = json["message"];
            response = message["content"];
        } else {
            response = "没有响应";
        }
        if(json.find("done") != json.end()) {
            done = json["done"];
        }
        // TODO: list
        return callback(response.c_str(), response.size(), done);
    });
}

static std::string body(const std::string& prompt, bool stream, const lifuren::RestChatOptions& options) {
    nlohmann::json body{};
    nlohmann::json messages{};
    messages.push_back({
        { "role",    "user" },
        { "content", prompt }
    });
    body["model"]    = options.model;
    body["stream"]   = stream;
    body["messages"] = messages;
    return body.dump();
}
