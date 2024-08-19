/**
 * 服务终端
 * 
 * 提供各种服务终端
 * 
 * @author acgist
 */

#ifndef LFR_HEADER_CLIENT_CLIENT_HPP
#define LFR_HEADER_CLIENT_CLIENT_HPP


#include <list>
#include <string>
#include <memory>
#include <functional>

#include "httplib.h"

#include "ClientOptions.hpp"

namespace lifuren {

/**
 * 终端父类
 */
class Client {

public:
    virtual ~Client();

};

/**
 * RAG查询器
 */
class RAGSearchEngine {

public:
    /**
     * 索引搜索
     * 
     * @param prompt 索引内容
     * 
     * @return 文档内容
     */
    virtual std::vector<std::string> search(const std::string& prompt) = 0;

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
    AuthType authType{ RestClient::AuthType::NONE };
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
     * @param config 配置
     * 
     * @return 是否成功
     */
    bool auth(const lifuren::config::RestConfig& config);

    /**
     * 授权
     * 
     * @param options 配置
     * 
     * @return 是否成功
     */
    bool auth(const lifuren::options::RestOptions& options);

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
    httplib::Result head(const std::string& path, const httplib::Headers& headers = {});

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
    httplib::Result putJson(const std::string& path, const std::string& data, const httplib::Headers& headers = {}) const;

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

    /**
     * 删除请求
     * 
     * @param path    请求地址
     * @param headers 请求头部
     * 
     * @return 响应内容
     */
    httplib::Result deletePath(const std::string& path, const httplib::Headers& headers = {});

};

/**
 * 命令终端
 */
class CommandClient : public Client {

};

/**
 * 聊天终端
 */
class ChatClient : public Client {

protected:
    // 历史聊天记录
    std::list<lifuren::chat::ChatMessage> historyMessages{};

public:
    // RAG查询器
    std::unique_ptr<lifuren::RAGSearchEngine> ragSearchEngine{ nullptr };

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
    /**
     * 重置会话
     * 
     * 1. 清除历史消息
     */
    void reset();

protected:
    /**
     * 添加历史消息
     * 
     * @param role    角色
     * @param message 消息内容
     * @param library 附加资料
     * @param done    是否完成
     */
    void appendMessage(const lifuren::chat::Role& role, const std::string& message, const std::vector<std::string>& library = {}, bool done = true);

};

/**
 * Ollama聊天终端
 * 
 * 项目地址：https://github.com/ollama/ollama
 */
class OllamaChatClient : public ChatClient {

public:
    // REST终端
    std::unique_ptr<lifuren::RestClient> restClient{ nullptr };
    // REST配置
    lifuren::options::RestChatOptions options{};

public:
    OllamaChatClient(lifuren::options::RestChatOptions options);
    ~OllamaChatClient();

public:
    std::string chat(const std::string& prompt) override;
    void chat(const std::string& prompt, std::function<bool(const char*, size_t, bool)> callback) override;

};

class OpenAIChatClient {
};

/**
 * 词嵌入终端
 */
class EmbeddingClient : public Client {

public:
    static std::unique_ptr<lifuren::EmbeddingClient> getClient(const std::string& embedding);

};

/**
 * CWV词嵌入终端
 * 
 * 项目地址：https://github.com/Embedding/Chinese-Word-Vectors
 */
class CWVEmbeddingClient : public EmbeddingClient {

};

class OllamaEmbeddingClient : public EmbeddingClient {
};

/**
 * 绘画终端
 */
class PaintClient : public Client {
};

class StableDiffusionCPPPaintClient {
};

} // END OF lifuren

#endif // END OF LFR_HEADER_CLIENT_CLIENT_HPP
