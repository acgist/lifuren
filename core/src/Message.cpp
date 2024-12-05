#include "lifuren/Message.hpp"

#include <map>

static std::map<lifuren::message::Type, std::function<void(bool, const char*)>> message_callback;

void lifuren::message::registerMessageCallback(lifuren::message::Type type, std::function<void(bool, const char*)> callback) {
    message_callback[type] = callback;
}

void lifuren::message::unregisterMessageCallback(lifuren::message::Type type) {
    message_callback.erase(type);
}

void lifuren::message::sendMessage(lifuren::message::Type type, bool finish, const char* message) {
    const auto iter = message_callback.find(type);
    if(iter == message_callback.end()) {
        return;
    }
    iter->second(finish, message);
}
