#include "../../header/REST.hpp"

httplib::Server lifuren::httpServer;

void lifuren::initHttpServer() {
    restGetIndex();
    SPDLOG_INFO("启动REST服务：{}", 8080);
    bool success = httpServer.listen("0.0.0.0", 8080);
    SPDLOG_INFO("结束REST服务：{} - {}", 8080, success);
}

void lifuren::shutdownHttpServer() {
    httpServer.stop();
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
