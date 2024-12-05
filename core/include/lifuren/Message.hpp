/**
 * 消息通知
 */
#ifndef LFR_HEADER_BOOT_MESSAGE_HPP
#define LFR_HEADER_BOOT_MESSAGE_HPP

#include <functional>

namespace lifuren::message {

// 消息类型
enum class Type {
};

extern void registerMessageCallback(Type type, std::function<void(bool, const char*)> callback);
extern void unregisterMessageCallback(Type type);
extern void sendMessage(Type type, bool finish, const char* message);

} // END OF lifuren::message

#endif // END OF LFR_HEADER_BOOT_MESSAGE_HPP
