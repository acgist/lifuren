#pragma once

#include <iostream>
#include <filesystem>

#include "glog/logging.h"

namespace lifuren {

/**
 * 加载GLog
 * 
 * @param argc 参数数量
 * @param argv 参数配置
 */
extern void init(int argc, char const* argv[]);

/**
 * 关闭GLog
 */
extern void shutdown();

}