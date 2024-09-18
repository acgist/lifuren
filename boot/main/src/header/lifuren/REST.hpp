/**
 * REST API
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_BOOT_REST_HPP
#define LFR_HEADER_BOOT_REST_HPP

#ifdef  _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include "nlohmann/json.hpp"

namespace httplib {
    class Server;
    class Response;
}

namespace lifuren {

namespace content::type {

    const char* const HTML = "text/html";
    const char* const ICON = "image/x-icon";
    const char* const JSON = "application/json";

}; // END OF content::type

// HTTP Server
extern httplib::Server httpServer;

// 加载HTTP服务
extern void initHttpServer();
// 关闭HTTP服务
extern void shutdownHttpServer();
/**
 * 成功响应
 * 
 * @param response 响应
 * @param body     内容
 */
extern void response(httplib::Response& response, const char* body);
/**
 * 响应
 * 
 * @param response 响应
 * @param code     响应编码
 * @param message  响应描述
 */
extern void response(httplib::Response& response, const char* code, const char* message);
/**
 * @param body 内容
 * 
 * @return 成功响应
 */
extern nlohmann::json buildResponse(const char* body);
/**
 * @param code    响应编码
 * @param message 响应描述
 * 
 * @return 响应
 */
extern nlohmann::json buildResponse(const char* code, const char* message);

// 公共接口
extern void restAPI();
// 图片接口
extern void restImageAPI();
// 诗词接口
extern void restPoetryAPI();

} // END lifuren

#endif // LFR_HEADER_BOOT_REST_HPP
