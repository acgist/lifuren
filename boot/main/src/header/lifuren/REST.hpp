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

#include "httplib.h"

#include "nlohmann/json.hpp"

namespace lifuren {

namespace content::type {

    const char* const HTML  = "text/html";
    const char* const JSON  = "application/json";
    const char* const EVENT = "text/event-stream";

};

// HTTP Server
extern httplib::Server httpServer;

// 加载HTTP服务
extern void initHttpServer();
// 关闭HTTP服务
extern void shutdownHttpServer();
// 响应
extern void response(httplib::Response& response, const char* body);
// 响应
extern void response(httplib::Response& response, const char* code, const char* message);

/**
 * @param message 响应消息
 * 
 * @return 响应
 */
extern nlohmann::json buildResponse(const char* body);

/**
 * @param code    响应编码
 * @param message 响应消息
 * 
 * @return 响应
 */
extern nlohmann::json buildResponse(const char* code, const char* message);

// 公共接口
extern void restAPI();
// 聊天
extern void restChatAPI();
// 图片
extern void restImageAPI();
// 诗词
extern void restPoetryAPI();

} // END lifuren

#endif // LFR_HEADER_BOOT_REST_HPP
