/**
 * Copyright(c) 2024-present acgist. ALl Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * REST API
 * 
 * 只提供生成接口不提供训练等其他接口
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

namespace httplib {
    
    class Server;
    class Response;

} // END OF httplib

namespace lifuren {

namespace content::type {

    const char* const HTML = "text/html";        // HTML
    const char* const ICON = "image/x-icon";     // ICON
    const char* const JSON = "application/json"; // JSON

}; // END OF content::type

extern httplib::Server httpServer; // HTTP Server

extern void initHttpServer();     // 加载HTTP服务
extern void shutdownHttpServer(); // 关闭HTTP服务

extern void restAPI();       // 公共接口
extern void restAudioAPI();  // 音频接口
extern void restImageAPI();  // 图片接口
extern void restVideoAPI();  // 视频接口
extern void restPoetryAPI(); // 诗词接口

/**
 * 成功响应
 */
extern void response(
    httplib::Response& response, // 响应
    const char* body // 响应内容
);

/**
 * 失败响应
 */
extern void response(
    httplib::Response& response, // 响应
    const char* code,   // 响应编码
    const char* message // 响应描述
);

} // END lifuren

#endif // LFR_HEADER_BOOT_REST_HPP
