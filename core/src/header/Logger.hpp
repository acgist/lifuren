/**
 * 日志工具
 * 
 * st：单线程版本（效率更高）
 * mt：多线程版本（线程安全）
 * 
 * @author acgist
 */
#pragma once

#include "spdlog/spdlog.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace lifuren {

namespace logger {

/**
 * 加载日志
 */
extern void init();

/**
 * 关闭日志
 */
extern void shutdown();

}

}
