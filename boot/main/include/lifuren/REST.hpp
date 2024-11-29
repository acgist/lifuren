/**
 * REST API
 * 
 * 只提供生成接口不提供训练等等其他接口
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_BOOT_REST_HPP
#define LFR_HEADER_BOOT_REST_HPP

#ifdef  _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <string>

namespace httplib {
    class Server;
    class Response;
}

namespace lifuren {

namespace content::type {

    const char* const HTML = "text/html";        // HTML
    const char* const ICON = "image/x-icon";     // ICON
    const char* const JSON = "application/json"; // JSON

}; // END OF content::type

extern httplib::Server httpServer; // HTTP Server

/**
 * 加载HTTP服务
 */
extern void initHttpServer();

/**
 * 关闭HTTP服务
 */
extern void shutdownHttpServer();

/**
 * 成功响应
 * 
 * @param response 响应
 * @param body     响应内容
 */
extern void response(httplib::Response& response, const char* body);

/**
 * 失败响应
 * 
 * @param response 响应
 * @param code     响应编码
 * @param message  响应描述
 */
extern void response(httplib::Response& response, const char* code, const char* message);

/**
 * @param body 响应内容
 * 
 * @return 成功响应
 */
extern std::string buildResponse(const char* body);

/**
 * @param code    响应编码
 * @param message 响应描述
 * 
 * @return 失败响应
 */
extern std::string buildResponse(const char* code, const char* message);

/**
 * 公共接口
 */
extern void restAPI();

/**
 * 音频接口
 */
extern void restAudioAPI();

/**
 * 图片接口
 */
extern void restImageAPI();

/**
 * 视频接口
 */
extern void restVideoAPI();

/**
 * 诗词接口
 */
extern void restPoetryAPI();

} // END lifuren

#endif // LFR_HEADER_BOOT_REST_HPP
