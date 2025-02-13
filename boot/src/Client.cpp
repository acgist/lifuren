#include "lifuren/Client.hpp"

#include "httplib.h"

#include "spdlog/spdlog.h"

static std::string      oauthToken  (const lifuren::RestClient* client, const std::string& path, const std::string& username, const std::string& password);
static httplib::Params  buildParams (const std::map<std::string, std::string>& params);
static httplib::Headers buildHeaders(const std::map<std::string, std::string>& headers);
static lifuren::RestClient::Response buildResponse(httplib::Result response);

lifuren::RestClient::RestClient(const std::string& baseUrl, bool trustAllCert, const std::string& certPath) : baseUrl(baseUrl) {
    this->client = std::make_unique<httplib::Client>(baseUrl);
    this->client->set_follow_location(true);
    #if defined(_DEBUG) || !defined(NDEBUG)
    this->client->set_read_timeout(15, 0);
    this->client->set_write_timeout(15, 0);
    this->client->set_connection_timeout(15, 0);
    #else
    this->client->set_read_timeout(30, 0);
    this->client->set_write_timeout(30, 0);
    this->client->set_connection_timeout(30, 0);
    #endif
    #ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if(trustAllCert) {
        this->client->enable_server_certificate_verification(true);
    } else if(!certPath.empty()) {
        this->client->set_ca_cert_path(certPath);
    } else {
        // -
    }
    #endif
    this->client->set_default_headers({
        { "User-Agent", "lifuren/1.0.0 (+https://gitee.com/acgist/lifuren)" }
    });
}

lifuren::RestClient::~RestClient() {
}

bool lifuren::RestClient::auth(const lifuren::config::RestConfig& config) {
    lifuren::RestClient::AuthType authType = lifuren::RestClient::AuthType::NONE;
    if(
        config.authType == "basic" ||
        config.authType == "Basic"
    ) {
        authType = lifuren::RestClient::AuthType::BASIC;
    } else if(
        config.authType == "token" ||
        config.authType == "Token" ||
        config.authType == "OAuth" ||
        config.authType == "oauth"
    ) {
        authType = lifuren::RestClient::AuthType::TOKEN;
    } else {
    }
    return this->auth(authType, config.username, config.password, config.authPath);
}

bool lifuren::RestClient::auth(
    const lifuren::RestClient::AuthType& authType,
    const std::string& username,
    const std::string& password,
    const std::string& path
) {
    this->authType = authType;
    this->username = username;
    this->password = password;
    if(authType == lifuren::RestClient::AuthType::NONE) {
        // -
    }  else if (authType == lifuren::RestClient::AuthType::BASIC) {
        this->client->set_basic_auth(username, password);
    } else if(authType == lifuren::RestClient::AuthType::TOKEN) {
        this->token     = oauthToken(this, path, username, password);
        this->tokenPath = path;
        if(this->token.empty()) {
            SPDLOG_WARN("Token授权失败：{}", path);
            return false;
        }
        SPDLOG_DEBUG("Token授权成功: {} - {}", path, this->token);
        this->client->set_default_headers({
            { "Authorization", "Bearer " + this->token }
        });
    }
    return true;
}

lifuren::RestClient::Response lifuren::RestClient::head(const std::string& path, const std::map<std::string, std::string>& headers) const {
    return buildResponse(this->client->Head(path, buildHeaders(headers)));
}

lifuren::RestClient::Response lifuren::RestClient::get(const std::string& path, const std::map<std::string, std::string>& headers) const {
    return buildResponse(this->client->Get(path, buildHeaders(headers)));
}

lifuren::RestClient::Response lifuren::RestClient::del(const std::string& path, const std::map<std::string, std::string>& headers) const {
    return buildResponse(this->client->Delete(path, buildHeaders(headers)));
}

lifuren::RestClient::Response lifuren::RestClient::putJson(const std::string& path, const std::string& data, const std::map<std::string, std::string>& headers) const {
    return buildResponse(this->client->Put(path, buildHeaders(headers), data, "application/json"));
}

lifuren::RestClient::Response lifuren::RestClient::putForm(const std::string& path, const std::string& data, const std::map<std::string, std::string>& headers) const {
    return buildResponse(this->client->Put(path, buildHeaders(headers), data, "application/x-www-form-urlencoded"));
}

lifuren::RestClient::Response lifuren::RestClient::putForm(const std::string& path, const std::map<std::string, std::string>& params, const std::map<std::string, std::string>& headers) const {
    return buildResponse(this->client->Put(path, buildHeaders(headers), buildParams(params)));
}

lifuren::RestClient::Response lifuren::RestClient::postJson(const std::string& path, const std::string& data, const std::map<std::string, std::string>& headers) const {
    return buildResponse(this->client->Post(path, buildHeaders(headers), data, "application/json"));
}

lifuren::RestClient::Response lifuren::RestClient::postForm(const std::string& path, const std::string& data, const std::map<std::string, std::string>& headers) const {
    return buildResponse(this->client->Post(path, buildHeaders(headers), data, "application/x-www-form-urlencoded"));
}

lifuren::RestClient::Response lifuren::RestClient::postForm(const std::string& path, const std::map<std::string, std::string>& params, const std::map<std::string, std::string>& headers) const {
    return buildResponse(this->client->Post(path, buildHeaders(headers), buildParams(params)));
}

bool lifuren::RestClient::postStream(const std::string& path, const std::string& data, std::function<bool(const char*, size_t)> callback, const std::map<std::string, std::string>& headers) const {
    httplib::Request request;
    request.path    = path;
    request.body    = data;
    request.method  = "POST";
    request.headers = buildHeaders(headers);
    request.set_header("Content-Type", "application/json");
    request.content_receiver = [callback](const char* data, size_t data_length, uint64_t /* offset */, uint64_t /* total_length */) {
        return callback(data, data_length);
    };
    return buildResponse(this->client->send(request));
}

static std::string oauthToken(const lifuren::RestClient* client, const std::string& path, const std::string& username, const std::string& password) {
    auto response = client->postForm(path, {
        { "username", username },
        { "password", password }
    });
    if(response) {
        return response.body;
    }
    return "";
}

static httplib::Params buildParams(const std::map<std::string, std::string>& params) {
    httplib::Params ret;
    if(params.empty()) {
        return ret;
    }
    for(const auto& [key, val] : params) {
        ret.emplace(key, val);
    }
    return ret;
}

static httplib::Headers buildHeaders(const std::map<std::string, std::string>& headers) {
    httplib::Headers ret;
    if(headers.empty()) {
        return ret;
    }
    for(const auto& [key, val] : headers) {
        ret.emplace(key, val);
    }
    return ret;
}

static lifuren::RestClient::Response buildResponse(httplib::Result response) {
    lifuren::RestClient::Response ret;
    if(response) {
        if(response->status >= httplib::StatusCode::BadRequest_400) {
            SPDLOG_WARN("RestClient失败响应：{} - {}", response->status, response->body);
            ret.success = false;
        } else {
            ret.success = true;
        }
        ret.status = response->status;
        ret.body   = std::move(response->body);
        for(auto& pair : response->headers) {
            ret.headers.emplace(std::move(pair.first), std::move(pair.second));
        }
    } else {
        SPDLOG_WARN("RestClient请求失败：{}", httplib::to_string(response.error()));
        ret.status  = 500;
        ret.success = false;
    }
    return ret;
}

lifuren::RestClient::Response::Response() {
}

lifuren::RestClient::Response::Response(const lifuren::RestClient::Response& response) {
    this->success = response.success;
    this->status  = response.status;
    this->headers = response.headers;
    this->body    = response.body;
}

lifuren::RestClient::Response::Response(const lifuren::RestClient::Response&& response) {
    this->success = response.success;
    this->status  = response.status;
    this->headers = std::move(response.headers);
    this->body    = std::move(response.body);
}

lifuren::RestClient::Response::~Response() {
}

lifuren::RestClient::Response::operator bool() const {
    return this->success;
}

std::string lifuren::http::toQuery(const std::map<std::string, std::string>& data) {
    std::string body;
    for(const auto& [key, val] : data) {
        body += key + "=" + httplib::detail::encode_query_param(val) + "&";
    }
    if(!body.empty()) {
        body.resize(body.size() - 1);
    }
    return body;
}
