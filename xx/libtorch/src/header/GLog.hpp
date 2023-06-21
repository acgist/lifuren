#pragma once

#include <filesystem>
#include "glog/logging.h"

/**
 * 日志
 */
namespace lifuren {

/**
 * 加载Glog
 * 
 * @param argc 参数数量
 * @param argv 参数值
 */
extern void initGlog(int argc, char const* argv[]);

/**
 * 关闭Glog
 */
extern void shutdownGlog();

}