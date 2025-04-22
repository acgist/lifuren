/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * CLI
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CLIENT_CLI_HPP
#define LFR_HEADER_CLIENT_CLI_HPP

namespace lifuren {

/**
 * @param argc 参数长度
 * @param argv 参数内容
 * 
 * @return 是否执行命令成功
 */
extern bool cli(const int argc, const char* const argv[]);

} // END OF lifuren

#endif // LFR_HEADER_CLIENT_CLI_HPP
