#include "lifuren/REST.hpp"

#if LFR_ENABLE_FLTK
#include "lifuren/FLTK.hpp"
#endif

#include "httplib.h"

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"

httplib::Server lifuren::restServer;

static bool restClose = false;

static void restHandler();
static void restGetIndex();
static void restGetFavicon();
static void restGetShutdown();

static std::string buildResponse(const char* body);
static std::string buildResponse(const char* code, const char* message);

void lifuren::initRestService() {
    restHandler();
    lifuren::restAPI();
    lifuren::restModelAPI();
    SPDLOG_INFO("启动REST服务：{} - {}", lifuren::config::restServerHost, lifuren::config::restServerPort);
    restServer.set_read_timeout(60);
    restServer.set_write_timeout(60);
    restServer.set_keep_alive_timeout(60);
    restServer.set_keep_alive_max_count(8);
    restServer.set_payload_max_length(128 * 1024 * 1024);
    const bool success = restServer.listen(lifuren::config::restServerHost, lifuren::config::restServerPort);
    restClose = true;
    SPDLOG_INFO("结束REST服务：{} - {}", lifuren::config::restServerHost, lifuren::config::restServerPort);
    #if LFR_ENABLE_FLTK
    lifuren::shutdownFltkService();
    #endif
}

void lifuren::shutdownRestService() {
    if(restClose) {
        return;
    }
    restClose = true;
    SPDLOG_INFO("关闭REST服务");
    restServer.stop();
}

void lifuren::response(httplib::Response& response, const char* body) {
    response.set_content(::buildResponse(body), lifuren::content::type::JSON);
}

void lifuren::response(httplib::Response& response, const char* code, const char* message) {
    response.set_content(::buildResponse(code, message), lifuren::content::type::JSON);
}

inline static std::string buildResponse(const char* body) {
    nlohmann::json response;
    response["header"] = {
        { "code",    "0000" },
        { "message", "成功" }
    };
    response["body"] = body;
    return response.dump();
}

inline static std::string buildResponse(const char* code, const char* message) {
    nlohmann::json response;
    response["header"] = {
        { "code",    code    },
        { "message", message }
    };
    return response.dump();
}

static void restHandler() {
    lifuren::restServer.set_error_handler([](const httplib::Request& request, httplib::Response& response) {
        SPDLOG_ERROR("系统错误：{} - {} - {} - {}", request.path, request.body, response.status, response.body);
        if(response.status == httplib::StatusCode::OK_200) {
            response.status = httplib::StatusCode::InternalServerError_500;
        }
        if(response.body.empty()) {
            response.set_content(::buildResponse(std::to_string(2000 + response.status).c_str(), "未知错误"), lifuren::content::type::JSON);
        }
    });
    lifuren::restServer.set_exception_handler([](const httplib::Request& request, httplib::Response& response, std::exception_ptr e) {
        SPDLOG_ERROR("系统异常：{} - {} - {} - {}", request.path, request.body, response.status, response.body);
        std::string message;
        try {
            std::rethrow_exception(e);
        } catch (std::exception& e) {
            message = ::buildResponse("9999", e.what());
        } catch (...) {
            message = ::buildResponse("9999", "未知错误");
        }
        response.status = httplib::StatusCode::InternalServerError_500;
        response.set_content(message, lifuren::content::type::JSON);
    });
}

void lifuren::restAPI() {
    restGetIndex();
    restGetFavicon();
    restGetShutdown();
}

static void restGetIndex() {
    lifuren::restServer.Get("/", [](const httplib::Request& /* request */, httplib::Response& response) {
        response.set_content(R"(<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>李夫人</title>
    <style type="text/css">
        p{text-align:center;}
        a{text-decoration:none;}
    </style>
</head>
<body>
    <p><a href="https://gitee.com/acgist/lifuren">李夫人</a></p>
    <p><a href="https://www.acgist.com">acgist</a></p>
</body>
</html>
)", lifuren::content::type::HTML);
    });
}

static void restGetFavicon() {
    lifuren::restServer.Get("/favicon.ico", [](const httplib::Request& /* request */, httplib::Response& response) {
        auto blob = lifuren::file::loadBlobFile(lifuren::config::baseFile("./favicon.ico"));
        if(blob.empty()) {
            return;
        }
        response.set_content(blob.data(), blob.size(), lifuren::content::type::ICON);
    });
}

static void restGetShutdown() {
    lifuren::restServer.Get("/shutdown", [](const httplib::Request& /* request */, httplib::Response& response) {
        response.set_content(::buildResponse("正在关机..."), lifuren::content::type::JSON);
        lifuren::shutdownRestService();
    });
}
