#include "lifuren/Logger.hpp"

#include <chrono>

#include "spdlog/spdlog.h"

#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

static size_t duration{ 0LL };

void lifuren::logger::init() {
    ::duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::vector<spdlog::sink_ptr> sinks{};
    // 开发日志
    #if defined(_DEBUG) || !defined(NDEBUG)
    sinks.reserve(2);
    auto stdoutColorSinkSPtr = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(stdoutColorSinkSPtr);
    #endif
    // 文件日志
    auto dailyFileSinkSPtr = std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/lifuren.log", 0, 0, false, 7);
    sinks.push_back(dailyFileSinkSPtr);
    // 默认日志
    auto logger = std::make_shared<spdlog::logger>("lifurenLogger", sinks.begin(), sinks.end());
    #if defined(_DEBUG) || !defined(NDEBUG)
    logger->set_level(spdlog::level::debug);
    #else
    logger->set_level(spdlog::level::info);
    #endif
    logger->flush_on(spdlog::level::warn);
    logger->set_pattern("[%D %T] [%L] [%t] [%s:%#] %v");
    spdlog::set_default_logger(logger);
    SPDLOG_DEBUG(R"(
        
        北方有佳人，绝世而独立。
        一顾倾人城，再顾倾人国。
        宁不知倾城与倾国，佳人难再得。
    )");
}

void lifuren::logger::shutdown() {
    const size_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    SPDLOG_INFO("持续时间：{}", (duration - ::duration));
    SPDLOG_DEBUG(R"(
        
        南有乔木，不可休思。汉有游女，不可求思。
        汉之广矣，不可泳思。江之永矣，不可方思。
    )");
    spdlog::drop_all();
    spdlog::shutdown();
}
