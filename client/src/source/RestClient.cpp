/**
 * https://github.com/yhirose/cpp-httplib
 * 
 * TODO:
 * 1. 优化拷贝性能
 */
#include "lifuren/Client.hpp"

#include <chrono>

#include "httplib.h"

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

static lifuren::RestClient::Response buildResponse(const httplib::Result& response);
static httplib::Params buildParams(const std::map<std::string, std::string>& params);
static httplib::Headers buildHeaders(const std::map<std::string, std::string>& headers);

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

lifuren::RestClient::Response lifuren::RestClient::head(const std::string& path, const std::map<std::string, std::string>& headers) {
    auto response = this->client->Head(path, buildHeaders(headers));
    return buildResponse(response);
}

lifuren::RestClient::Response lifuren::RestClient::get(const std::string& path, const std::map<std::string, std::string>& headers) const {
    auto response = this->client->Get(path, buildHeaders(headers));
    return buildResponse(response);
}

lifuren::RestClient::Response lifuren::RestClient::putJson(const std::string& path, const std::string& data, const std::map<std::string, std::string>& headers) const {
    auto response = this->client->Put(path, buildHeaders(headers), data, "application/json");
    return buildResponse(response);
}

lifuren::RestClient::Response lifuren::RestClient::postJson(const std::string& path, const std::string& data, const std::map<std::string, std::string>& headers) const {
    auto response = this->client->Post(path, buildHeaders(headers), data, "application/json");
    return buildResponse(response);
}

lifuren::RestClient::Response lifuren::RestClient::postForm(const std::string& path, const std::string& data, const std::map<std::string, std::string>& headers) const {
    auto response = this->client->Post(path, buildHeaders(headers), data, "application/x-www-form-urlencoded");
    return buildResponse(response);
}

lifuren::RestClient::Response lifuren::RestClient::post(const std::string& path, const std::map<std::string, std::string>& params, const std::map<std::string, std::string>& headers) const {
    auto response = this->client->Post(path, buildHeaders(headers), buildParams(params));
    return buildResponse(response);
}

bool lifuren::RestClient::postStream(const std::string& path, const std::string& data, std::function<bool(const char*, size_t)> callback, const std::map<std::string, std::string>& headers) const {
    httplib::Request request{};
    request.path    = path;
    request.body    = data;
    request.method  = "POST";
    request.headers = buildHeaders(headers);
    request.set_header("Content-Type", "application/json");
    request.content_receiver = [&callback](const char* data, size_t data_length, uint64_t /* offset */, uint64_t /* total_length */) {
        return callback(data, data_length);
    };
    auto response = this->client->send(request);
    return buildResponse(response);
}

lifuren::RestClient::Response lifuren::RestClient::deletePath(const std::string& path, const std::map<std::string, std::string>& headers) {
    auto response = this->client->Delete(path, buildHeaders(headers));
    return buildResponse(response);
}

static std::string oauthToken(const lifuren::RestClient& client, const std::string& path, const std::string& username, const std::string& password) {
    auto response = client.post(path, {
        { "username", username },
        { "password", password }
    });
    if(response) {
        return response.body;
    }
    return "";
}

static lifuren::RestClient::Response buildResponse(const httplib::Result& response) {
    lifuren::RestClient::Response ret;
    if(response) {
        ret.status = response->status;
        ret.body   = response->body;
        for(const auto& pair : response->headers) {
            ret.headers.emplace(pair.first, pair.second);
        }
        if(
            response->status != httplib::StatusCode::OK_200 &&
            response->status != httplib::StatusCode::Created_201
        ) {
            SPDLOG_DEBUG("RestClient响应失败：{} - {}", response->status, response->body);
            ret.success = false;
        } else {
            ret.success = true;
        }
    } else {
        SPDLOG_DEBUG("RestClient请求失败：{}", httplib::to_string(response.error()));
        ret.status  = 500;
        ret.success = false;
    }
    return ret;
}

static httplib::Params buildParams(const std::map<std::string, std::string>& params) {
    httplib::Params ret{};
    if(params.empty()) {
        return ret;
    }
    for(const auto& pair : params) {
        ret.emplace(pair.first, pair.second);
    }
    return ret;
}

static httplib::Headers buildHeaders(const std::map<std::string, std::string>& headers) {
    httplib::Headers ret{};
    if(headers.empty()) {
        return ret;
    }
    for(const auto& pair : headers) {
        ret.emplace(pair.first, pair.second);
    }
    return ret;
}
