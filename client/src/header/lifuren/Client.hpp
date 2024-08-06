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

#include "httplib.h"

namespace lifuren {

class Client;
class RestClient;
class ChatClient;
class EmbeddingClient;
class RAGClient;
class PaintClient;
class VideoClient;

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
class RestClient {

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
    bool postStream(const std::string& path, const std::string& data, std::function<bool(const char* data, size_t data_length)> callback, const httplib::Headers& headers = {}) const;

};

}

#endif // END OF LFR_HEADER_CLIENT_CLIENT_HPP