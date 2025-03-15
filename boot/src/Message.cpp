#include "lifuren/Message.hpp"

#include <map>
#include <thread>

#include "spdlog/spdlog.h"

static std::map<std::thread::id, std::function<void(const char*)>> message_callback;

void lifuren::message::registerMessageCallback(std::function<void(const char*)> callback) {
    message_callback[std::this_thread::get_id()] = callback;
    SPDLOG_DEBUG("注册消息通知回调：{}", message_callback.size());
}

void lifuren::message::unregisterMessageCallback() {
    message_callback.erase(std::this_thread::get_id());
    SPDLOG_DEBUG("取消消息通知回调：{}", message_callback.size());
}

void lifuren::message::sendMessage(const char* message) {
    auto iterator = message_callback.find(std::this_thread::get_id());
    if(iterator == message_callback.end()) {
        return;
    }
    iterator->second(message);
}
