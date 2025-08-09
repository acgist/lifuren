#include "lifuren/Logger.hpp"

#include <chrono>

#include "opencv2/core/utils/logger.hpp"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

static size_t duration{ 0 }; // 系统运行持续时间

void lifuren::logger::init() {
    ::duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::vector<spdlog::sink_ptr> sinks{};
    #if defined(_DEBUG) || !defined(NDEBUG)
    sinks.reserve(2);
    auto stdoutColorSinkSPtr = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(stdoutColorSinkSPtr);
    #else
    sinks.reserve(1);
    #endif
    auto dailyFileSinkSPtr = std::make_shared<spdlog::sinks::daily_file_sink_mt>("./logs/lifuren.log", 0, 0, false, 7);
    sinks.push_back(dailyFileSinkSPtr);
    auto logger = std::make_shared<spdlog::logger>("lifuren-logger", sinks.begin(), sinks.end());
    #if defined(_DEBUG) || !defined(NDEBUG)
    logger->set_level(spdlog::level::debug);
    #else
    logger->set_level(spdlog::level::info);
    #endif
    logger->flush_on(spdlog::level::info);
    logger->set_pattern("[%m-%d %H:%M:%S.%e] [%L] [%6t] [%-8!s:%4#] %v");
    spdlog::set_default_logger(logger);
    SPDLOG_DEBUG(R"(
        
        北方有佳人，绝世而独立。
        一顾倾人城，再顾倾人国。
        宁不知倾城与倾国，佳人难再得。
    )");
}

void lifuren::logger::opencv::init() {
    #if defined(_DEBUG) || !defined(NDEBUG)
    cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_INFO);
    #else
    cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_INFO);
    #endif
}

void lifuren::logger::stop() {
    const size_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    SPDLOG_INFO("持续时间：{}", (duration - ::duration));
    SPDLOG_DEBUG(R"(
        
        中庭地白树栖鸦，冷露无声湿桂花。
        今夜月明人尽望，不知秋思落谁家。
    )");
    spdlog::drop_all();
    spdlog::shutdown();
}
