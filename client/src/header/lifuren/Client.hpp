/**
 * 服务终端
 * 
 * 提供各种服务终端
 * 
 * @author acgist
 */

#ifndef LFR_HEADER_CLIENT_CLIENT_HPP
#define LFR_HEADER_CLIENT_CLIENT_HPP

#include <map>
#include <list>
#include <cstdio>
#include <string>
#include <memory>
#include <functional>

#include "lifuren/config/Config.hpp"

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
 * RAG查询器
 */
class RAGSearchClient {

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

/**
 * 命令终端
 */
class CommandClient : public Client {

public:
    CommandClient(const std::string& command, std::function<void(bool, const std::string&)> callback = nullptr);
    ~CommandClient();

protected:
    // 结束状态
    int code = -1;
    // 执行结果
    std::string result;
    // 执行命令
    std::string command;
    // 命令管道
    FILE* pipe{ nullptr };
    // 回调函数
    std::function<void(bool, const std::string&)> callback{ nullptr };

public:
    const int& execute();
    void shutdown() const;
    const int& getCode() const;
    const std::string& getResult() const;

};

/**
 * 词嵌入终端
 */
class EmbeddingClient : public Client {

public:
    static std::unique_ptr<lifuren::EmbeddingClient> getClient(const std::string& embedding);

};

class OllamaEmbeddingClient : public EmbeddingClient {

};

/**
 * Chinese-Word-Vectors词嵌入终端
 * 
 * 项目地址：https://github.com/Embedding/Chinese-Word-Vectors
 */
class ChineseWordVectorsEmbeddingClient : public EmbeddingClient {

};

/**
 * 绘画终端
 */
class PaintClient : public Client {

public:
struct PaintOptions {
    std::string image;
    std::string video;
    std::string model;
    std::string prompt;
    std::string output;
    
    size_t seed   = 42;
    size_t steps  = 30;
    size_t width  = 512;
    size_t height = 512;

    bool color = true;
};

public:
/**
 * 绘画回调
 * 
 * @param finish  是否完成
 * @param percent 进度
 * @param message 没有完成=提示内容、任务完成=图片路径
 * 
 * @return 是否结束
 */
using PaintCallback = std::function<bool(bool finish, float percent, const std::string& message)>;

protected:
    PaintCallback callback{ nullptr };

public:
    PaintClient(PaintCallback callback = nullptr);
    ~PaintClient();

public:
    /**
     * @param options  提示内容
     * @param callback 消息回调
     * 
     * @return 是否成功
     */
    virtual bool paint(const PaintOptions& options, PaintCallback callback = nullptr) = 0;

};

class CycleGANPaintClient {
    // TODO: 实现算法
};

class StyleGANPaintClient {
    // TODO: 实现算法
};

/**
 * StableDiffusionCPP终端
 * 
 * https://github.com/leejet/stable-diffusion.cpp
 */
class StableDiffusionCPPPaintClient : public PaintClient {

private:

public:
    StableDiffusionCPPPaintClient();
    ~StableDiffusionCPPPaintClient();

public:
    bool paint(const PaintOptions& options, PaintClient::PaintCallback callback = nullptr) override;

};

/**
 * 诗词终端
 */
class PoetizeClient : public Client {

};

class RNNPoetizeClient : public PoetizeClient {

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CLIENT_CLIENT_HPP
