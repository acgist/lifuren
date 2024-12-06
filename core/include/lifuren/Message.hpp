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
    NONE,

};

extern thread_local lifuren::message::Type thread_message_type;

extern void registerMessageCallback  (Type type, std::function<void(bool, const char*)> callback);
extern void unregisterMessageCallback(Type type);
extern void sendMessage(           const char* message, bool finish = false);
extern void sendMessage(Type type, const char* message, bool finish = false);

} // END OF lifuren::message

#endif // END OF LFR_HEADER_BOOT_MESSAGE_HPP
