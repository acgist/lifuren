/**
 * CLI
 * 
 * ./lifuren[.exe] poetize prompt
 * ./lifuren[.exe] embedding
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_BOOT_CLI_HPP
#define LFR_HEADER_BOOT_CLI_HPP

namespace lifuren {

/**
 * @param argc 参数数量
 * @param argv 命令参数
 * 
 * @return 是否执行命令
 */
extern bool cli(const int argc, const char* const argv[]);

} // END OF lifuren

#endif // LFR_HEADER_BOOT_CLI_HPP
