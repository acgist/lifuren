#include "lifuren/Message.hpp"

#include "spdlog/spdlog.h"

static std::function<void(const char*)> message_callback { nullptr };

void lifuren::message::register_message_callback(std::function<void(const char*)> callback) {
    SPDLOG_DEBUG("注册消息通知回调");
    message_callback = callback;
}

void lifuren::message::unregister_message_callback() {
    SPDLOG_DEBUG("取消消息通知回调");
    message_callback = nullptr;
}

void lifuren::message::send_message(const char* message) {
    if(message_callback) {
        message_callback(message);
    }
}
