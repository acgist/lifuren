#include "lifuren/CLI.hpp"

#include <string>
#include <vector>

#include "spdlog/spdlog.h"

bool lifuren::cli(const int argc, const char* const argv[]) {
    if(argc <= 1) {
        return false;
    }
    for(int i = 0; i < argc; ++i) {
        SPDLOG_DEBUG("命令参数：{}", argv[i]);
    }
    const char* const command = argv[0];
    if(command == "poetize") {
        if(argc <= 2) {
            SPDLOG_WARN("缺少提示内容");
            return true;
        }
        std::vector<std::string> prompt;
        for(int i = 2; i < argc; ++i) {
            prompt.push_back(argv[i]);
        }
        // TODO: 实现
    } else if(command == "embedding") {
        // TODO: 实现
    } else {
        SPDLOG_WARN("不支持的命令：{}", command);
    }
    return true;
}
