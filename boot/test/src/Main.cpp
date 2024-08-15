/**
 * 测试代码
 * 
 * @author acgist
 */
#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
