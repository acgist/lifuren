/**
 * 消息通知
 */
#ifndef LFR_HEADER_BOOT_MESSAGE_HPP
#define LFR_HEADER_BOOT_MESSAGE_HPP

#include <functional>

#ifndef MESSAGE_LIMITS
#define MESSAGE_CONSOLE    0
#define MESSAGE_AUDIO_MIN  1000
#define MESSAGE_VIDEO_MIN  2000
#define MESSAGE_POETRY_MIN 3000
#define MESSAGE_MAX        9999
#endif

namespace lifuren::message {

// 消息类型
enum class Type {
    
    CLI_CONSOLE             = MESSAGE_CONSOLE,
    AUDIO_EMBEDDING         = MESSAGE_AUDIO_MIN,
    AUDIO_MODEL_TRAIN       = MESSAGE_AUDIO_MIN + 1,
    AUDIO_MODEL_PRED        = MESSAGE_AUDIO_MIN + 2,
    VIDEO_EMBEDDING         = MESSAGE_VIDEO_MIN,
    VIDEO_MODEL_TRAIN       = MESSAGE_VIDEO_MIN + 1,
    VIDEO_MODEL_PRED        = MESSAGE_VIDEO_MIN + 2,
    POETRY_EMBEDDING_PEPPER = MESSAGE_POETRY_MIN,
    POETRY_EMBEDDING_POETRY = MESSAGE_POETRY_MIN + 1,
    POETRY_MODEL_TRAIN      = MESSAGE_POETRY_MIN + 2,
    POETRY_MODEL_PRED       = MESSAGE_POETRY_MIN + 3,
    NONE                    = MESSAGE_MAX,

};

extern thread_local lifuren::message::Type thread_message_type;

extern void registerMessageCallback  (Type type, std::function<void(bool, const char*)> callback);
extern void unregisterMessageCallback(Type type);
extern void sendMessage(           const char* message, bool finish = false);
extern void sendMessage(Type type, const char* message, bool finish = false);

} // END OF lifuren::message

#endif // END OF LFR_HEADER_BOOT_MESSAGE_HPP
