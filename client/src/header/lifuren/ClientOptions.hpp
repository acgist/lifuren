/**
 * 终端配置
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CLIENT_CLIENTOPTIONS_HPP
#define LFR_HEADER_CLIENT_CLIENTOPTIONS_HPP

#include <string>
#include <vector>

#include "lifuren/config/Config.hpp"

namespace lifuren {

namespace chat {

/**
 * 角色
 */
enum class Role {

    USER,
    ASSISTANT,

};

/**
 * 聊天信息
 */
struct ChatMessage {

    // 橘色
    Role role;
    // 消息
    std::string message;
    // 附加资料
    std::vector<std::string> library;
    // 是否完成
    bool done;

};

} // END OF chat

namespace options {

struct RestOptions {

    // 地址
    std::string api;
    // 账号
    std::string username;
    // 密码
    std::string password;
    // 授权类型
    std::string authType;
    // 授权地址
    std::string authPath;

};

struct LLMOptions {

    double topP;
    size_t topK;
    double temperature;
    std::string options;

};

struct ChatOptions : LLMOptions {

    // 聊天模型
    std::string model;

};

struct RestChatOptions : RestOptions, ChatOptions {

    // 聊天地址
    std::string path;

    void of(const lifuren::config::OllamaConfig& config) {
        this->api   = config.api;
        this->path  = config.chatClient.path;
        this->model = config.chatClient.model;
    }

};

struct EmbeddingOptions {

    // 模型
    std::string model;

};

struct RestEmbeddingOptions : RestOptions, EmbeddingOptions {

    std::string path;

};

/**
 * 标记配置
 */
struct Mark {

};

/**
 * 图片标记页面配置
 */
struct ImageMark : Mark {
};

/**
 * 诗词标记页面配置
 */
struct PoetryMark : Mark {
};

/**
 * 文档标记页面配置
 */
struct DocumentMark : Mark {
};

struct ImageOptions {
};

struct PoetryOptions {
};

} // END OF options

} // END OF lifuren

#endif // END OF LFR_HEADER_CLIENT_CLIENTOPTIONS_HPP