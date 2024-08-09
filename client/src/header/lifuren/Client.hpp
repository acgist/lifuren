/**
 * 服务终端
 * 
 * 提供各种服务终端
 * 
 * @author acgist
 */

#ifndef LFR_HEADER_CLIENT_CLIENT_HPP
#define LFR_HEADER_CLIENT_CLIENT_HPP


#include <string>
#include <memory>
#include <functional>

#include "httplib.h"

#include "ClientOptions.hpp"

namespace lifuren {

class Client;
class RestClient;
class ChatClient;
class RestChatClient;
class LocalChatClient;
class EmbeddingClient;
class RestEmbeddingClient;
class LocalEmbeddingClient;
class RAGClient;
class PaintClient;
class RestPaintClient;
class LocalPaintClient;
class VideoClient;
class RestVideoClient;
class LocalVideoClient;

/**
 * 终端父类
 */
class Client {

public:
    virtual ~Client();

};

/**
 * REST终端
 */
class RestClient : public Client {

/**
 * 授权方式
 */
enum class AuthType {

    NONE,  // 没有授权
    BASIC, // Basic
    TOKEN, // OAuth Bearer Token

};

public:
    // 基础地址
    std::string baseUrl;
    // HTTP终端
    std::unique_ptr<httplib::Client> client{ nullptr };
    // 授权方式
    AuthType authType{ lifuren::RestClient::AuthType::NONE };
    // 账号
    std::string username;
    // 密码
    std::string password;
    // Token
    std::string token;
    // Token地址
    std::string tokenPath;

public:
    /**
     * @param baseUrl      基础地址
     * @param trustAllCert 信任所有证书
     * @param certPath     证书地址
     */
    RestClient(const std::string& baseUrl, bool trustAllCert = false, const std::string& certPath = "./ca.crt");
    virtual ~RestClient();

public:
    /**
     * 授权
     * 
     * @param authType 授权方式
     * @param username 账号
     * @param password 密码
     * @param path     Token请求地址
     * 
     * @return 是否成功
     */
    bool auth(const AuthType& authType, const std::string& username, const std::string& password, const std::string& path = "");

    /**
     * @param path    请求地址
     * @param headers 请求头部
     * 
     * @return 响应内容
     */
    httplib::Result get(const std::string& path, const httplib::Headers& headers = {}) const;

    /**
     * 数据请求
     * 
     * @param path    请求地址
     * @param data    请求数据
     * @param headers 请求头部
     * 
     * @return 响应内容
     */
    httplib::Result postJson(const std::string& path, const std::string& data, const httplib::Headers& headers = {}) const;

    /**
     * 表单请求
     * 
     * @param path    请求地址
     * @param data    请求数据
     * @param headers 请求头部
     * 
     * @return 响应内容
     */
    httplib::Result postForm(const std::string& path, const std::string& data, const httplib::Headers& headers = {}) const;

    /**
     * 表单请求
     * 
     * @param path    请求地址
     * @param params  请求参数
     * @param headers 请求头部
     * 
     * @return 响应内容
     */
    httplib::Result post(const std::string& path, const httplib::Params& params, const httplib::Headers& headers = {}) const;

    /**
     * 流式请求
     * 
     * @param path     请求地址
     * @param data     请求数据
     * @param callback 响应回调
     * @param headers  请求头部
     * 
     * @return 是否成功
     */
    bool postStream(const std::string& path, const std::string& data, std::function<bool(const char*, size_t)> callback, const httplib::Headers& headers = {}) const;

};

/**
 * 聊天终端
 */
class ChatClient : public Client {

protected:
    // TODO: 历史记录
    // std::list

public:
    /**
     * 正常聊天
     * 
     * @param prompt 提示词
     * 
     * @return 返回内容
     */
    virtual std::string chat(const std::string& prompt) = 0;
    /**
     * 流式聊天
     * 
     * @param prompt   提示词
     * @param callback 回调函数
     */
    virtual void chat(const std::string& prompt, std::function<bool(const char*, size_t, bool)> callback) = 0;

};

/**
 * Rest聊天终端
 */
class RestChatClient : public ChatClient {

public:
    // REST终端
    std::unique_ptr<lifuren::RestClient> restClient{ nullptr };
    // 配置
    lifuren::RestChatOptions options{};

public:
    RestChatClient(lifuren::RestChatOptions options);
    ~RestChatClient();

public:
    std::string chat(const std::string& prompt) override;
    void chat(const std::string& prompt, std::function<bool(const char*, size_t, bool)> callback) override;

};

/**
 * 词嵌入终端
 */
class EmbeddingClient : public Client {

};

class RestEmbeddingClient : public EmbeddingClient {

};

/**
 * RAG终端
 * 
 * 文档解析、文档分段、文档搜索
 */
class RAGClient : public Client {

public:
    // 词嵌入终端
    std::unique_ptr<EmbeddingClient> embeddingClient{ nullptr };

};

/**
 * 绘画终端
 */
class PaintClient : public Client {
};

/**
 * 视频终端
 */
class VideoClient : public Client {
};

} // END OF lifuren

#endif // END OF LFR_HEADER_CLIENT_CLIENT_HPP