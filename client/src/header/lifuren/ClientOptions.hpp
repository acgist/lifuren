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
struct MarkOptions {

};

/**
 * 图片标记页面配置
 */
struct ImageMarkOptions : MarkOptions {
};

/**
 * 诗词标记页面配置
 */
struct PoetryMarkOptions : MarkOptions {
};

struct ImageOptions {
};

struct PoetryOptions {
};

} // END OF options

} // END OF lifuren

#endif // END OF LFR_HEADER_CLIENT_CLIENTOPTIONS_HPP