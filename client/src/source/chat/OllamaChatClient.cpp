#include "lifuren/Client.hpp"

#include <cstring>

#include "nlohmann/json.hpp"

static std::string promptLibrary(const std::string& prompt, const std::vector<std::string>& library);
static std::string body(const std::string& prompt, bool stream, const lifuren::options::RestChatOptions& options);

lifuren::OllamaChatClient::OllamaChatClient(lifuren::options::RestChatOptions options) : options(options) {
    this->restClient = std::make_unique<lifuren::RestClient>(options.api);
}

lifuren::OllamaChatClient::~OllamaChatClient() {
}

std::string lifuren::OllamaChatClient::chat(const std::string& prompt) {
    std::vector<std::string> library{};
    if(this->ragSearchEngine) {
        library = this->ragSearchEngine->search(prompt);
    }
    this->appendMessage(lifuren::chat::Role::USER, prompt, library);
    auto response = this->restClient->postJson(this->options.path, body(promptLibrary(prompt, library), false, this->options));
    if(!response) {
        return "请求错误";
    }
    auto json = nlohmann::json::parse(response->body);
    if(json.find("error") != json.end()) {
        return json["error"].template get<std::string>();
    }
    if(json.find("message") != json.end()) {
        auto& message = json["message"];
        std::string content = message["content"];
        this->appendMessage(lifuren::chat::Role::ASSISTANT, content);
        return content;
    }
    return "没有响应";
}

void lifuren::OllamaChatClient::chat(const std::string& prompt, std::function<bool(const char*, size_t, bool)> callback) {
    std::vector<std::string> library{};
    if(this->ragSearchEngine) {
        library = this->ragSearchEngine->search(prompt);
    }
    this->appendMessage(lifuren::chat::Role::USER, prompt, library);
    bool success = this->restClient->postStream(this->options.path, body(promptLibrary(prompt, library), true, this->options), [this, &callback](const char* text, size_t length) {
        auto json = nlohmann::json::parse(std::string(text, length));
        bool done = true;
        std::string content;
        if(json.find("done") != json.end()) {
            done = json["done"];
        }
        if(json.find("error") != json.end()) {
            content = json["error"].template get<std::string>();
        } else if(json.find("message") != json.end()) {
            auto& message = json["message"];
            content = message["content"];
            this->appendMessage(lifuren::chat::Role::ASSISTANT, content, {}, done);
        } else {
            content = "没有响应";
        }
        return callback(content.c_str(), content.size(), done);
    });
    if(!success) {
        const char* response = "请求失败";
        callback(response, std::strlen(response), true);
    }
}

static std::string promptLibrary(const std::string& prompt, const std::vector<std::string>& library) {
    std::string ret = prompt;
    if(library.empty()) {
        return ret;
    }
    for(auto& value : library) {
        ret += "\n" + value;
    }
    return ret;
}

static std::string body(const std::string& prompt, bool stream, const lifuren::options::RestChatOptions& options) {
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
