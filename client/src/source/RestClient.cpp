/**
 * https://github.com/yhirose/cpp-httplib
 */
#include "lifuren/Client.hpp"

#include "spdlog/spdlog.h"

/**
 * 授权请求
 * 
 * @param client   请求终端
 * @param path     请求地址
 * @param username 账号
 * @param password 密码
 */
static std::string oauthToken(const lifuren::RestClient& client, const std::string& path, const std::string& username, const std::string& password);

lifuren::RestClient::RestClient(const std::string& baseUrl, bool trustAllCert, const std::string& certPath) : baseUrl(baseUrl) {
    this->client = std::make_unique<httplib::Client>(baseUrl);
    #ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if(trustAllCert) {
        this->client->enable_server_certificate_verification(true);
    } else {
        this->client->set_ca_cert_path(certPath);
    }
    #endif
    // 默认头部
    // this->client->set_default_headers({
    //     { "key", "value" }
    // });
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
    if(response) {
        if(response->status != httplib::StatusCode::OK_200) {
            SPDLOG_DEBUG("Rest Client Get请求失败：{} - {}", response->status, response->body);
        } else {
            // 忽略
        }
    } else {
        auto error = response.error();
        SPDLOG_DEBUG("Rest Client Get请求失败：{} - {}", response->status, httplib::to_string(error));
    }
    return response;
}

httplib::Result lifuren::RestClient::postJson(const std::string& path, const std::string& data, const httplib::Headers& headers) const {
    auto response = this->client->Post(path, headers, data, "application/json");
    if(response) {
        if(response->status != httplib::StatusCode::OK_200) {
            SPDLOG_DEBUG("Rest Client PostJson请求失败：{} - {}", response->status, response->body);
        } else {
            // 忽略
        }
    } else {
        auto error = response.error();
        SPDLOG_DEBUG("Rest Client PostJson请求失败：{} - {}", response->status, httplib::to_string(error));
    }
    return response;
}

httplib::Result lifuren::RestClient::postForm(const std::string& path, const std::string& data, const httplib::Headers& headers) const {
    auto response = this->client->Post(path, headers, data, "application/x-www-form-urlencoded");
    if(response) {
        if(response->status != httplib::StatusCode::OK_200) {
            SPDLOG_DEBUG("Rest Client PostForm请求失败：{} - {}", response->status, response->body);
        } else {
            // 忽略
        }
    } else {
        auto error = response.error();
        SPDLOG_DEBUG("Rest Client PostForm请求失败：{} - {}", response->status, httplib::to_string(error));
    }
    return response;
}

httplib::Result lifuren::RestClient::post(const std::string& path, const httplib::Params& params, const httplib::Headers& headers) const {
    auto response = this->client->Post(path, headers, params);
    if(response) {
        if(response->status != httplib::StatusCode::OK_200) {
            SPDLOG_DEBUG("Rest Client Post请求失败：{} - {}", response->status, response->body);
        } else {
            // 忽略
        }
    } else {
        auto error = response.error();
        SPDLOG_DEBUG("Rest Client Post请求失败：{} - {}", response->status, httplib::to_string(error));
    }
    return response;
}

static std::string oauthToken(const lifuren::RestClient& client, const std::string& path, const std::string& username, const std::string& password) {
    httplib::Params params{
        { "username", username },
        { "password", password }
    };
    auto response = client.post(path, params);
    if(response && response->status == httplib::StatusCode::OK_200) {
        return response->body;
    }
    return "";
}
