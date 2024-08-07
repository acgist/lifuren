/**
 * REST API
 * 
 * REST.hpp必须在FLTK.hpp的前面
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_BOOT_REST_HPP
#define LFR_HEADER_BOOT_REST_HPP

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

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

extern void restGetImageToImage();
extern void restGetLabelToImage();
extern void restGetPoetryToImage();

extern void restGetImageToPoetry();
extern void restGetLabelToPoetry();

extern void restGetVideoToVideo();

extern void restGetDocsIndex();
extern void restPostDocsIndex();
extern void restGetChatSetting();
extern void restPostChatSetting();
extern void restGetChat();

} // END lifuren

#endif // LFR_HEADER_BOOT_REST_HPP
