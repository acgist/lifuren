#include "lifuren/Message.hpp"

#include <map>
#include <thread>

#include "spdlog/spdlog.h"

static std::function<void(const char*)> message_callback { nullptr };

void lifuren::message::registerMessageCallback(std::function<void(const char*)> callback) {
    SPDLOG_DEBUG("注册消息通知回调");
    message_callback = callback;
}

void lifuren::message::unregisterMessageCallback() {
    SPDLOG_DEBUG("取消消息通知回调");
    message_callback = nullptr;
}

void lifuren::message::sendMessage(const char* message) {
    if(message_callback) {
        message_callback(message);
    }
}
