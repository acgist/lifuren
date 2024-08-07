/**
 * 终端配置
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CLIENT_CLIENTOPTIONS_HPP
#define LFR_HEADER_CLIENT_CLIENTOPTIONS_HPP

#include <string>

namespace lifuren {

struct RestOptions {

    // 地址
    std::string api;
    // 账号
    std::string username;
    // 密码
    std::string password;
    // 授权类型
    std::string authType;

};

struct LLMOptions {

    double topP;
    size_t topK;
    double temperature;
    std::string options;

};

struct ChatOptions : public LLMOptions {

    // 聊天模型
    std::string model;

};

struct RestChatOptions : public RestOptions, public ChatOptions {

    // 聊天地址
    std::string path;

};

struct EmbeddingOptions {

    // 模型
    std::string model;

};

struct RestEmbeddingOptions : public RestOptions, public EmbeddingOptions {

    std::string path;

};

/**
 * 数据集配置
 */
struct Dataset {

    // 数据集目录
    std::string path;

};

/**
 * 文档标记页面配置
 */
struct DocsMark : public Dataset {
};

/**
 * 图片标记页面配置
 */
struct ImageMark : public Dataset {
};

/**
 * 诗词标记页面配置
 */
struct PoetryMark : public Dataset {
};

struct ImageOptions {
};

struct VideoOptions {
};

} // END OF lifuren

#endif // END OF LFR_HEADER_CLIENT_CLIENTOPTIONS_HPP