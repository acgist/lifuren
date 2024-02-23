/**
 * 日志
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
