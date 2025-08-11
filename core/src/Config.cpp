#include "lifuren/Config.hpp"

#include <mutex>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <filesystem>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"

std::string lifuren::config::base_dir = "";

static const char* const EMPTY_CHARS = " \t\r\n"; // 空白字符

/**
 * @param value 字符串
 * 
 * @return 去掉空格后的字符串
 */
inline std::string trim(const std::string& value) {
    std::size_t index = value.find_first_not_of(EMPTY_CHARS);
    if(index == std::string::npos) {
        return {};
    }
    std::size_t jndex = value.find_last_not_of(EMPTY_CHARS);
    return value.substr(index, jndex - index + 1);
}

void lifuren::config::init(const int argc, const char* const argv[]) {
    if(argc > 0) {
        lifuren::config::base_dir = std::filesystem::absolute(std::filesystem::path(argv[0]).parent_path()).string();
    }
    SPDLOG_DEBUG("项目启动绝对路径：{}", lifuren::config::base_dir);
}

std::string lifuren::config::baseFile(const std::string& path) {
    return lifuren::file::join({ lifuren::config::base_dir, path }).string();
}

size_t lifuren::config::uuid() noexcept(true) {
    auto timePoint = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(timePoint);
    auto localtime = std::localtime(&timestamp);
    int i = 0;
    {
              static int index     = 0;
        const static int MIN_INDEX = 0;
        const static int MAX_INDEX = 100000;
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        i = index;
        if(++index >= MAX_INDEX) {
            index = MIN_INDEX;
        }
    }
    size_t id = 1000000000000000LL * (localtime->tm_year - 100) + // + 1900 - 2000
                10000000000000LL   * (localtime->tm_mon  +   1) +
                100000000000LL     *  localtime->tm_mday        +
                1000000000LL       *  localtime->tm_hour        +
                10000000LL         *  localtime->tm_min         +
                100000LL           *  localtime->tm_sec         +
                i;
    return id;
}
