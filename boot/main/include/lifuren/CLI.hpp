/**
 * CLI API
 * 
 * ./lifuren[.exe] 命令 [参数...]
 * ./lifuren[.exe] act       [train|pred] [model|dataset] file
 * ./lifuren[.exe] paint     [train|pred] [model|dataset] file
 * ./lifuren[.exe] compose   [train|pred] [model|dataset] file
 * ./lifuren[.exe] poetize   [train|pred] [model|dataset] prompt
 * quantization
 * ./lifuren[.exe] pcm       path
 * ./lifuren[.exe] pepper    path
 * ./lifuren[.exe] embedding path
 * ./lifuren[.exe] [?|help]
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_BOOT_CLI_HPP
#define LFR_HEADER_BOOT_CLI_HPP

namespace lifuren {

/**
 * 命令执行
 * 
 * 执行命令行时不会启动FLTK和HTTP
 * 
 * @param argc 参数数量
 * @param argv 命令参数
 * 
 * @return 是否执行命令
 */
extern bool cli(const int argc, const char* const argv[]);

} // END OF lifuren

#endif // LFR_HEADER_BOOT_CLI_HPP
