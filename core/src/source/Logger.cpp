#include "../header/Logger.hpp"

void lifuren::logger::init() {
    std::vector<spdlog::sink_ptr> sinks{};
    // 开发日志
    #if defined(__DEBUG__)
    auto stdoutSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(stdoutSink);
    #endif
    // 文件日志
    auto dailyFileSink = std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/lifuren.log", 0, 0);
    sinks.push_back(dailyFileSink);
    // 默认日志
    auto logger = std::make_shared<spdlog::logger>("lifurenLogger", sinks.begin(), sinks.end());
    logger->set_level(spdlog::level::debug);
    logger->set_pattern("[%D %T] [%L] [%t] [%s:%#] %v");
    spdlog::set_default_logger(logger);
    SPDLOG_DEBUG(R"(
        
        北方有佳人，绝世而独立。
        一顾倾人城，再顾倾人国。
        宁不知倾城与倾国，佳人难再得。
    )");
}

void lifuren::logger::shutdown() {
    SPDLOG_DEBUG("关闭日志");
    spdlog::drop_all();
    spdlog::shutdown();
}
