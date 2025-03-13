#include "lifuren/Message.hpp"

#include <map>

#include "spdlog/spdlog.h"

// 消息类型 = 消息回调
static std::function<void(bool, const char*)> message_callback;

void lifuren::message::registerMessageCallback(std::function<void(bool, const char*)> callback) {
    SPDLOG_DEBUG("注册消息回调");
    message_callback = callback;
}

void lifuren::message::unregisterMessageCallback() {
    SPDLOG_DEBUG("取消消息回调注册");
    message_callback = nullptr;
}

void lifuren::message::sendMessage(const char* message, bool finish) {
    if(message_callback) {
        message_callback(finish, message);
    } else {
        // -
    }
}
