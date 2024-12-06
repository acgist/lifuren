/**
 * 消息通知
 */
#ifndef LFR_HEADER_BOOT_MESSAGE_HPP
#define LFR_HEADER_BOOT_MESSAGE_HPP

#include <functional>

namespace lifuren::message {

// 消息类型
enum class Type {

    AUDIO_AUDIO_FILE_TO_PCM_FILE,

};

extern void registerMessageCallback(Type type, std::function<void(bool, const char*)> callback);
extern void unregisterMessageCallback(Type type);
extern void sendMessage(Type type, bool finish, const char* message);

} // END OF lifuren::message

#endif // END OF LFR_HEADER_BOOT_MESSAGE_HPP
