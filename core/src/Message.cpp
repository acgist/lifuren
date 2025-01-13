#include "lifuren/Message.hpp"

#include <map>
#include <thread>

#include "spdlog/spdlog.h"

thread_local lifuren::message::Type lifuren::message::thread_message_type = lifuren::message::Type::NONE;

static std::map<lifuren::message::Type, std::function<void(bool, const char*)>> message_callback;

void lifuren::message::registerMessageCallback(lifuren::message::Type type, std::function<void(bool, const char*)> callback) {
    SPDLOG_DEBUG("注册消息回调：{}", static_cast<int>(type));
    thread_message_type    = type;
    message_callback[type] = callback;
}

void lifuren::message::unregisterMessageCallback(lifuren::message::Type type) {
    message_callback.erase(type);
    thread_message_type = lifuren::message::Type::NONE;
    SPDLOG_DEBUG("取消消息回调：{}", static_cast<int>(type));
}

void lifuren::message::sendMessage(const char* message, bool finish) {
    lifuren::message::sendMessage(thread_message_type, message, finish);
}

void lifuren::message::sendMessage(lifuren::message::Type type, const char* message, bool finish) {
    const auto iter = message_callback.find(type);
    if(iter == message_callback.end()) {
        return;
    }
    iter->second(finish, message);
}
