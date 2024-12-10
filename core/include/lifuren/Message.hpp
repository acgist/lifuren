/**
 * 消息通知
 */
#ifndef LFR_HEADER_BOOT_MESSAGE_HPP
#define LFR_HEADER_BOOT_MESSAGE_HPP

#include <functional>

#ifndef MESSAGE_LIMITS
#define MESSAGE_AUDIO_MIN  1000
#define MESSAGE_IMAGE_MIN  2000
#define MESSAGE_VIDEO_MIN  3000
#define MESSAGE_POETRY_MIN 4000
#define MESSAGE_MAX        9999
#endif

namespace lifuren::message {

// 消息类型
enum class Type {
    
    AUDIO_AUDIO_FILE_TO_PCM_FILE  = MESSAGE_AUDIO_MIN,
    IMAGE_AUDIO_FILE_TO_PCM_FILE  = MESSAGE_IMAGE_MIN,
    VIDEO_AUDIO_FILE_TO_PCM_FILE  = MESSAGE_VIDEO_MIN,
    POETRY_AUDIO_FILE_TO_PCM_FILE = MESSAGE_POETRY_MIN,
    NONE = MESSAGE_MAX,

};

extern thread_local lifuren::message::Type thread_message_type;

extern void registerMessageCallback  (Type type, std::function<void(bool, const char*)> callback);
extern void unregisterMessageCallback(Type type);
extern void sendMessage(           const char* message, bool finish = false);
extern void sendMessage(Type type, const char* message, bool finish = false);

} // END OF lifuren::message

#endif // END OF LFR_HEADER_BOOT_MESSAGE_HPP
