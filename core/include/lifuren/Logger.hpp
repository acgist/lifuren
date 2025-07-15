/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 日志
 * 
 * st：单线程版本（效率更高）
 * mt：多线程版本（线程安全）
 * 
 * https://fmt.dev/latest/
 * https://fmt.dev/latest/api/
 * https://fmt.dev/latest/syntax/
 * https://github.com/gabime/spdlog/
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_LOGGER_HPP
#define LFR_HEADER_CORE_LOGGER_HPP

namespace lifuren::logger {

extern void init(); // 加载日志
extern void stop(); // 关闭日志

namespace opencv {

extern void init(); // 加载OpenCV日志

} // END OF opencv

} // END OF lifuren::logger

#endif // LFR_HEADER_CORE_LOGGER_HPP
