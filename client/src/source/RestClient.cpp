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
    // 超时时间
    this->client->set_read_timeout(30, 0);
    this->client->set_write_timeout(30, 0);
    this->client->set_connection_timeout(30, 0);
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
    }
    return true;
}

httplib::Result lifuren::RestClient::get(const std::string& path, const httplib::Headers& headers) const {
    auto response = this->client->Get(path, headers);
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
    return true;
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
        if(response->status != httplib::StatusCode::OK_200) {
            SPDLOG_DEBUG("RestClient响应失败：{} - {}", response->status, response->body);
        } else {
            return true;
        }
    } else {
        SPDLOG_DEBUG("RestClient请求失败：{} - {}", response->status, httplib::to_string(response.error()));
    }
    return false;
}
