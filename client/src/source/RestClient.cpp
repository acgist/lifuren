/**
 * https://github.com/yhirose/cpp-httplib
 */
#include "lifuren/Client.hpp"

#include <chrono>

#include "spdlog/spdlog.h"

/**
 * 授权请求
 * 
 * @param client   请求终端
 * @param path     请求地址
 * @param username 账号
 * @param password 密码
 * 
 * @return Token
 */
static std::string oauthToken(const lifuren::RestClient& client, const std::string& path, const std::string& username, const std::string& password);

/**
 * 验证响应
 * 
 * @param response 响应信息
 * 
 * @return 是否成功
 */
static bool checkResponse(const httplib::Result& response);

lifuren::RestClient::RestClient(const std::string& baseUrl, bool trustAllCert, const std::string& certPath) : baseUrl(baseUrl) {
    this->client = std::make_unique<httplib::Client>(baseUrl);
    this->client->set_follow_location(true);
    // 超时时间
    #if defined(_DEBUG) || !defined(NDEBUG)
    this->client->set_read_timeout(5, 0);
    this->client->set_write_timeout(5, 0);
    this->client->set_connection_timeout(5, 0);
    #else
    this->client->set_read_timeout(30, 0);
    this->client->set_write_timeout(30, 0);
    this->client->set_connection_timeout(30, 0);
    #endif
    // HTTPS
    #ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if(trustAllCert) {
        this->client->enable_server_certificate_verification(true);
    } else {
        this->client->set_ca_cert_path(certPath);
    }
    #endif
    // 默认头部
    this->client->set_default_headers({
        { "User-Agent", "lifuren/1.0.0 (+https://gitee.com/acgist/lifuren)" }
    });
}

lifuren::RestClient::~RestClient() {
}

bool lifuren::RestClient::auth(const lifuren::config::RestConfig& config) {
    lifuren::RestClient::AuthType authType = lifuren::RestClient::AuthType::NONE;
    if(config.authType == "Basic") {
        authType = lifuren::RestClient::AuthType::BASIC;
    } else if(config.authType == "Token") {
        authType = lifuren::RestClient::AuthType::TOKEN;
    } else {

    }
    return this->auth(authType, config.username, config.password, config.authPath);
}

bool lifuren::RestClient::auth(const lifuren::options::RestOptions& options) {
    lifuren::RestClient::AuthType authType = lifuren::RestClient::AuthType::NONE;
    if(options.authType == "Basic") {
        authType = lifuren::RestClient::AuthType::BASIC;
    } else if(options.authType == "Token") {
        authType = lifuren::RestClient::AuthType::TOKEN;
    } else {

    }
    return this->auth(authType, options.username, options.password, options.authPath);
}

bool lifuren::RestClient::auth(const lifuren::RestClient::AuthType& authType, const std::string& username, const std::string& password, const std::string& path) {
    this->authType = authType;
    this->username = username;
    this->password = password;
    if(authType == lifuren::RestClient::AuthType::NONE) {
    }  else if (authType == lifuren::RestClient::AuthType::BASIC) {
        this->client->set_basic_auth(username, password);
    } else if(authType == lifuren::RestClient::AuthType::TOKEN) {
        this->token     = oauthToken(*this, path, username, password);
        this->tokenPath = path;
        SPDLOG_DEBUG("RestClient Token: {}", this->token);
        if(this->token.empty()) {
            return false;
        }
        this->client->set_default_headers({
            { "Authorization", "Bearer " + this->token }
        });
    }
    return true;
}

httplib::Result lifuren::RestClient::head(const std::string& path, const httplib::Headers& headers) {
    auto response = this->client->Head(path, headers);
    checkResponse(response);
    return response;
}

httplib::Result lifuren::RestClient::get(const std::string& path, const httplib::Headers& headers) const {
    auto response = this->client->Get(path, headers);
    checkResponse(response);
    return response;
}

httplib::Result lifuren::RestClient::putJson(const std::string& path, const std::string& data, const httplib::Headers& headers) const {
    auto response = this->client->Put(path, headers, data, "application/json");
    checkResponse(response);
    return response;
}

httplib::Result lifuren::RestClient::postJson(const std::string& path, const std::string& data, const httplib::Headers& headers) const {
    auto response = this->client->Post(path, headers, data, "application/json");
    checkResponse(response);
    return response;
}

httplib::Result lifuren::RestClient::postForm(const std::string& path, const std::string& data, const httplib::Headers& headers) const {
    auto response = this->client->Post(path, headers, data, "application/x-www-form-urlencoded");
    checkResponse(response);
    return response;
}

httplib::Result lifuren::RestClient::post(const std::string& path, const httplib::Params& params, const httplib::Headers& headers) const {
    auto response = this->client->Post(path, headers, params);
    checkResponse(response);
    return response;
}

bool lifuren::RestClient::postStream(const std::string& path, const std::string& data, std::function<bool(const char*, size_t)> callback, const httplib::Headers& headers) const {
    httplib::Request request{};
    request.path    = path;
    request.body    = data;
    request.method  = "POST";
    request.headers = headers;
    request.set_header("Content-Type", "application/json");
    request.content_receiver = [&callback](const char* data, size_t data_length, uint64_t /* offset */, uint64_t /* total_length */) {
        return callback(data, data_length);
    };
    auto response = this->client->send(request);
    checkResponse(response);
    return response;
}

httplib::Result lifuren::RestClient::deletePath(const std::string& path, const httplib::Headers& headers) {
    auto response = this->client->Delete(path, headers);
    checkResponse(response);
    return response;
}

static std::string oauthToken(const lifuren::RestClient& client, const std::string& path, const std::string& username, const std::string& password) {
    auto response = client.post(path, {
        { "username", username },
        { "password", password }
    });
    if(response && response->status == httplib::StatusCode::OK_200) {
        return response->body;
    }
    return "";
}

static bool checkResponse(const httplib::Result& response) {
    if(response) {
        if(
            response->status != httplib::StatusCode::OK_200 &&
            response->status != httplib::StatusCode::Created_201
        ) {
            SPDLOG_DEBUG("RestClient响应失败：{} - {}", response->status, response->body);
        } else {
            return true;
        }
    } else {
        SPDLOG_DEBUG("RestClient请求失败：{}", httplib::to_string(response.error()));
    }
    return false;
}
