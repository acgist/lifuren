#include "lifuren/REST.hpp"

#if LFR_ENABLE_FLTK
#include "lifuren/FLTK.hpp"
#endif

#include "httplib.h"

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"

httplib::Server lifuren::httpServer;

/**
 * @return 成功响应
 */
inline static std::string buildResponse(
    const char* body // 响应内容
);

/**
 * @return 失败响应
 */
inline static std::string buildResponse(
    const char* code,   // 响应编码
    const char* message // 响应描述
);

// 是否关闭
static bool restClose = false;

static void restHandler();
static void restGetIndex();
static void restGetFavicon();
static void restGetShutdown();

void lifuren::initHttpServer() {
    restHandler();
    lifuren::restAPI();
    lifuren::restAudioAPI();
    lifuren::restImageAPI();
    lifuren::restVideoAPI();
    lifuren::restPoetryAPI();
    SPDLOG_INFO("启动REST服务：{} - {}", lifuren::config::httpServerHost.c_str(), lifuren::config::httpServerPort);
    const bool success = httpServer.listen(lifuren::config::httpServerHost.c_str(), lifuren::config::httpServerPort);
    // httpServer.set_read_timeout(60);
    // httpServer.set_write_timeout(60);
    // httpServer.set_keep_alive_timeout(60);
    SPDLOG_INFO("REST服务已经结束：{} - {}", lifuren::config::httpServerHost.c_str(), lifuren::config::httpServerPort);
    #if LFR_ENABLE_FLTK
    lifuren::shutdownFltkWindow();
    #endif
}

void lifuren::shutdownHttpServer() {
    if(restClose) {
        SPDLOG_INFO("REST服务已经结束");
        return;
    }
    restClose = true;
    SPDLOG_INFO("结束REST服务：{} - {}", lifuren::config::httpServerHost.c_str(), lifuren::config::httpServerPort);;
    httpServer.stop();
}

void lifuren::response(httplib::Response& response, const char* body) {
    response.set_content(::buildResponse(body), lifuren::content::type::JSON);
}

void lifuren::response(httplib::Response& response, const char* code, const char* message) {
    response.set_content(::buildResponse(code, message), lifuren::content::type::JSON);
}

inline static std::string buildResponse(const char* body) {
    nlohmann::json response;
    nlohmann::json header;
    header["code"]     = "0000";
    header["message"]  = "成功";
    response["header"] = header;
    response["body"]   = body;
    return response.dump();
}

inline static std::string buildResponse(const char* code, const char* message) {
    nlohmann::json response;
    nlohmann::json header;
    header["code"]     = code;
    header["message"]  = message;
    response["header"] = header;
    return response.dump();
}

static void restHandler() {
    lifuren::httpServer.set_error_handler([](const httplib::Request& request, httplib::Response& response) {
        SPDLOG_ERROR("发生系统错误：{} - {} - {}", request.path, request.body, response.status);
        response.set_content(::buildResponse("9999", "未知错误"), lifuren::content::type::JSON);
    });
    lifuren::httpServer.set_exception_handler([](const httplib::Request& request, httplib::Response& response, std::exception_ptr e) {
        SPDLOG_ERROR("发生系统异常：{} - {} - {}", request.path, request.body, response.status);
        std::string message{};
        try {
            std::rethrow_exception(e);
        } catch (std::exception& e) {
            message = ::buildResponse("9999", e.what());
        } catch (...) {
            message = ::buildResponse("9999", "未知异常");
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
    lifuren::httpServer.Get("/", [](const httplib::Request& request, httplib::Response& response) {
        response.set_content(R"(
<!DOCTYPE html>
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
    lifuren::httpServer.Get("/favicon.ico", [](const httplib::Request& request, httplib::Response& response) {
        char * data  { nullptr };
        size_t length{ 0LL };
        lifuren::file::loadFile(lifuren::config::baseFile("./favicon.ico"), &data, length);
        if(length == 0LL) {
            return;
        }
        response.set_content(data, length, lifuren::content::type::ICON);
        delete[] data;
        data = nullptr;
    });
}

static void restGetShutdown() {
    lifuren::httpServer.Get("/shutdown", [](const httplib::Request& request, httplib::Response& response) {
        response.set_content(::buildResponse("0000", "正在关机"), lifuren::content::type::JSON);
        lifuren::shutdownHttpServer();
    });
}
