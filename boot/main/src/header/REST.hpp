/**
 * REST API
 * 
 * REST.hpp必须在FLTK.hpp的前面
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_BOOT_REST_HPP
#define LFR_HEADER_BOOT_REST_HPP

#include "httplib.h"

#include "nlohmann/json.hpp"

namespace lifuren {

// HTTP Server
extern httplib::Server httpServer;

// 加载HTTP服务
extern void initHttpServer();

// 关闭HTTP服务
extern void shutdownHttpServer();

/**
 * @param code    响应编码
 * @param message 响应消息
 * 
 * @return 响应
 */
extern nlohmann::json buildResponse(const char* code, const char* message);

// 主页
extern void restGetIndex();

// 关闭
extern void restGetShutdown();

} // END lifuren

#endif // LFR_HEADER_BOOT_REST_HPP
