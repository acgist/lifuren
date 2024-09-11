/**
 * 服务终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CLIENT_CLIENT_HPP
#define LFR_HEADER_CLIENT_CLIENT_HPP

#include <map>
#include <list>
#include <mutex>
#include <cstdio>
#include <string>
#include <memory>
#include <functional>

#include "lifuren/Config.hpp"

namespace httplib {

class Client;

};

namespace lifuren {

/**
 * 终端父类
 */
class Client {

public:
    Client();
    virtual ~Client();

};

/**
 * 有状态的终端
 */
class StatefulClient {

protected:
    // 是否运行
    bool running{ false };
    // 状态锁
    std::mutex mutex;

public:
    const bool& isRunning() const;
    void changeState();
    void changeState(bool running);

public:
    StatefulClient();
    virtual ~StatefulClient();

};

/**
 * RAG搜索终端
 */
class RAGSearchClient {

public:
    /**
     * @param prompt 搜索内容
     * @param size   结果数量
     * 
     * @return 文档内容
     */
    virtual std::vector<std::string> search(const std::string& prompt, const int size = 4) = 0;

};

/**
 * REST终端
 */
class RestClient : public Client {

public:

/**
 * 响应内容
 */
class Response {

public:
    int  status  = 200;
    bool success = true;
    std::map<std::string, std::string> headers;
    std::string body;

public:
    Response();
    Response(const Response&  response);
    Response(const Response&& response);
    ~Response();

public:
    operator bool();

};

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
    Response head(const std::string& path, const std::map<std::string, std::string>& headers = {});

    /**
     * @param path    请求地址
     * @param headers 请求头部
     * 
     * @return 响应内容
     */
    Response get(const std::string& path, const std::map<std::string, std::string>& headers = {}) const;

    /**
     * 数据请求
     * 
     * @param path    请求地址
     * @param data    请求数据
     * @param headers 请求头部
     * 
     * @return 响应内容
     */
    Response putJson(const std::string& path, const std::string& data, const std::map<std::string, std::string>& headers = {}) const;

    /**
     * 数据请求
     * 
     * @param path    请求地址
     * @param data    请求数据
     * @param headers 请求头部
     * 
     * @return 响应内容
     */
    Response postJson(const std::string& path, const std::string& data, const std::map<std::string, std::string>& headers = {}) const;

    /**
     * 表单请求
     * 
     * @param path    请求地址
     * @param data    请求数据
     * @param headers 请求头部
     * 
     * @return 响应内容
     */
    Response postForm(const std::string& path, const std::string& data, const std::map<std::string, std::string>& headers = {}) const;

    /**
     * 表单请求
     * 
     * @param path    请求地址
     * @param params  请求参数
     * @param headers 请求头部
     * 
     * @return 响应内容
     */
    Response post(const std::string& path, const std::map<std::string, std::string>& params, const std::map<std::string, std::string>& headers = {}) const;

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
    bool postStream(const std::string& path, const std::string& data, std::function<bool(const char*, size_t)> callback, const std::map<std::string, std::string>& headers = {}) const;

    /**
     * 删除请求
     * 
     * @param path    请求地址
     * @param headers 请求头部
     * 
     * @return 响应内容
     */
    Response deletePath(const std::string& path, const std::map<std::string, std::string>& headers = {});

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CLIENT_CLIENT_HPP
