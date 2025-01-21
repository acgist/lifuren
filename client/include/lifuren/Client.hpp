/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 终端
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CLIENT_CLIENT_HPP
#define LFR_HEADER_CLIENT_CLIENT_HPP

#include <map>
#include <tuple>
#include <memory>
#include <string>
#include <functional>

#include "lifuren/Config.hpp"

namespace httplib {

class Client;

} // END OF httplib

namespace lifuren {

/**
 * 模型终端
 * 
 * @param C 模型配置
 * @param I 模型输入
 * @param O 模型输出
 */
template<typename C, typename I, typename O>
class ModelClient {

public:
    virtual bool save(const std::string& path = "./", const std::string& filename = "lifuren.pt") = 0; // 保存模型
    virtual bool load(const std::string& path = "./", const std::string& filename = "lifuren.pt") = 0; // 加载模型
    virtual bool trainValAndTest(C params, const bool val = true, const bool test = true)         = 0; // 训练模型
    virtual std::tuple<bool, O> pred(const I& input) = 0; // 预测结果

};

/**
 * 模型终端
 * 
 * @param C 模型配置
 * @param I 模型输入
 * @param O 模型输出
 * @param M 模型实现
 */
template<typename C, typename I, typename O, typename M>
class ModelImplClient : public ModelClient<C, I, O> {

protected:
    std::unique_ptr<M> model{ nullptr }; // 模型实现

public:
    virtual bool save(const std::string& path = "./", const std::string& filename = "lifuren.pt") override;
    virtual bool load(const std::string& path = "./", const std::string& filename = "lifuren.pt") override;
    virtual bool trainValAndTest(C params, const bool val = true, const bool test = true)         override;
    virtual std::tuple<bool, O> pred(const I& input) = 0;

};

/**
 * REST终端
 */
class RestClient {

public:
/**
 * 授权方式
 */
enum class AuthType {

    NONE,  // 没有授权
    BASIC, // Basic
    TOKEN, // OAuth Bearer Token

};

/**
 * 响应内容
 */
class Response {

public:
    bool success = true; // 是否成功
    int  status  = 200;  // 响应码
    std::string body;    // 响应体
    std::map<std::string, std::string> headers; // 响应头

public:
    Response();
    Response(const Response&  response);
    Response(const Response&& response);
    virtual ~Response();

public:
    operator bool() const;

};

public:
    std::string baseUrl;   // 基础地址
    std::string username;  // 账号
    std::string password;  // 密码
    std::string token;     // Token
    std::string tokenPath; // Token请求地址
    AuthType authType{ RestClient::AuthType::NONE };    // 授权方式
    std::unique_ptr<httplib::Client> client{ nullptr }; // HTTP Client

public:
    RestClient(
        const std::string& baseUrl  = "",        // 基础地址
        bool  trustAllCert          = false,     // 信任所有证书
        const std::string& certPath = "./ca.crt" // CA证书地址
    );
    virtual ~RestClient();

public:
    /**
     * @return 是否成功
     */
    bool auth(
        const lifuren::config::RestConfig& config // 配置
    );

    /**
     * @return 是否成功
     */
    bool auth(
        const AuthType   & authType, // 授权方式
        const std::string& username, // 账号
        const std::string& password, // 密码
        const std::string& path = "" // Token请求地址
    );

    /**
     * @return 响应内容
     */
    Response head(
        const std::string& path, // 请求地址
        const std::map<std::string, std::string>& headers = {} // 请求头部
    ) const;

    /**
     * @return 响应内容
     */
    Response get(
        const std::string& path, // 请求地址
        const std::map<std::string, std::string>& headers = {} // 请求头部
    ) const;

    /**
     * @return 响应内容
     */
    Response del(
        const std::string& path, // 请求地址
        const std::map<std::string, std::string>& headers = {} // 请求头部
    ) const;

    /**
     * @return 响应内容
     */
    Response putJson(
        const std::string& path, // 请求地址
        const std::string& data, // 请求数据
        const std::map<std::string, std::string>& headers = {} // 请求头部
    ) const;

    /**
     * @return 响应内容
     */
    Response putForm(
        const std::string& path, // 请求地址
        const std::string& data, // 请求数据
        const std::map<std::string, std::string>& headers = {} // 请求头部
    ) const;

    /**
     * @return 响应内容
     */
    Response putForm(
        const std::string& path, // 请求地址
        const std::map<std::string, std::string>& params,       // 请求参数
        const std::map<std::string, std::string>& headers = {}  // 请求头部
    ) const;

    /**
     * @return 响应内容
     */
    Response postJson(
        const std::string& path, // 请求地址
        const std::string& data, // 请求数据
        const std::map<std::string, std::string>& headers = {} // 请求头部
    ) const;

    /**
     * @return 响应内容
     */
    Response postForm(
        const std::string& path, // 请求地址
        const std::string& data, // 请求数据
        const std::map<std::string, std::string>& headers = {} // 请求头部
    ) const;

    /**
     * @return 响应内容
     */
    Response postForm(
        const std::string& path, // 请求地址
        const std::map<std::string, std::string>& params,       // 请求参数
        const std::map<std::string, std::string>& headers = {}  // 请求头部
    ) const;

    /**
     * @return 是否成功
     */
    bool postStream(
        const std::string& path, // 请求地址
        const std::string& data, // 请求数据
        std::function<bool(const char*, size_t)> callback,     // 响应回调
        const std::map<std::string, std::string>& headers = {} // 请求头部
    ) const;

};

namespace http {

/**
 * @return 请求参数
 */
extern std::string toQuery(
    const std::map<std::string, std::string>& data // 请求数据
);

} // END OF http

} // END OF lifuren

template<typename C, typename I, typename O, typename M>
bool lifuren::ModelImplClient<C, I, O, M>::save(const std::string& path, const std::string& filename) {
    if(!this->model) {
        return false;
    }
    return this->model->save(path, filename);
}

template<typename C, typename I, typename O, typename M>
bool lifuren::ModelImplClient<C, I, O, M>::load(const std::string& path, const std::string& filename) {
    if(!this->model) {
        return false;
    }
    return this->model->load(path, filename);
}

template<typename C, typename I, typename O, typename M>
bool lifuren::ModelImplClient<C, I, O, M>::trainValAndTest(C params, const bool val, const bool test) {
    if(!this->model) {
        this->model = std::make_unique<M>(params);
    }
    if(this->model) {
        this->model->trainValAndTest(val, test);
        return true;
    } else {
        return false;
    }
}

#endif // END OF LFR_HEADER_CLIENT_CLIENT_HPP
