#include "../../header/REST.hpp"
#if __FLTK__
#include "../../header/FLTK.hpp"
#endif

#include "Logger.hpp"
#include "config/Config.hpp"

httplib::Server lifuren::httpServer;

// 是否关闭
static bool restClose = false;

void lifuren::initHttpServer() {
    restGetIndex();
    restGetShutdown();
    SPDLOG_INFO("启动REST服务：{}", lifuren::config::httpServerPort);
    bool success = httpServer.listen(lifuren::config::httpServerHost.c_str(), lifuren::config::httpServerPort);
    SPDLOG_INFO("结束REST服务：{} - {}", lifuren::config::httpServerPort, success);
    #if __FLTK__
    lifuren::shutdownFltkWindow();
    #endif
}

void lifuren::shutdownHttpServer() {
    if(restClose) {
        return;
    }
    restClose = true;
    httpServer.stop();
}

nlohmann::json lifuren::buildResponse(const char* code, const char* message) {
    nlohmann::json response;
    nlohmann::json header;
    header["code"]     = code;
    header["message"]  = message;
    response["header"] = header;
    return response;
}

void lifuren::restGetIndex() {
    httpServer.Get("/", [](const httplib::Request& request, httplib::Response& response) {
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
        )", "text/html");
    });
}

void lifuren::restGetShutdown() {
    httpServer.Get("/shutdown", [](const httplib::Request& request, httplib::Response& response) {
        response.set_content(buildResponse("0000", "正在关机").dump(), "application/json");
        lifuren::shutdownHttpServer();
    });
}
