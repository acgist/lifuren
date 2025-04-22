#include "lifuren/Logger.hpp"

#include <chrono>

#include "spdlog/spdlog.h"

#include "lifuren/Message.hpp"

#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

static size_t duration{ 0 }; // 系统运行持续时间

// 消息日志
template<typename M>
class MessageLogger : public spdlog::sinks::base_sink<M> {

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t buf;
        spdlog::sinks::base_sink<M>::formatter_->format(msg, buf);
        std::string message;
        message.resize(buf.size());
        std::copy_n(buf.data(), buf.size(), message.data());
        lifuren::message::sendMessage(message.data());
    }

    void flush_() override {
    }

};

using message_sink_mt = MessageLogger<std::mutex>;
using message_sink_st = MessageLogger<spdlog::details::null_mutex>;

void lifuren::logger::init() {
    ::duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::vector<spdlog::sink_ptr> sinks{};
    #if defined(_DEBUG) || !defined(NDEBUG)
    sinks.reserve(3);
    auto stdoutColorSinkSPtr = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(stdoutColorSinkSPtr);
    #else
    sinks.reserve(2);
    #endif
    // 文件日志
    auto dailyFileSinkSPtr = std::make_shared<spdlog::sinks::daily_file_sink_mt>("./logs/lifuren.log", 0, 0, false, 7);
    sinks.push_back(dailyFileSinkSPtr);
    // 消息日志
    auto messageSinkSPtr = std::make_shared<message_sink_mt>();
    sinks.push_back(messageSinkSPtr);
    // 默认日志
    auto logger = std::make_shared<spdlog::logger>("lifuren-logger", sinks.begin(), sinks.end());
    #if defined(_DEBUG) || !defined(NDEBUG)
    logger->set_level(spdlog::level::debug);
    #else
    logger->set_level(spdlog::level::info);
    #endif
    logger->flush_on(spdlog::level::warn);
    logger->set_pattern("[%m-%d %H:%M:%S.%e] [%L] [%6t] [%-8!s:%4#] %v");
    spdlog::set_default_logger(logger);
    SPDLOG_DEBUG(R"(
        
        北方有佳人，绝世而独立。
        一顾倾人城，再顾倾人国。
        宁不知倾城与倾国，佳人难再得。
    )");
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
