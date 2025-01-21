/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 消息通知
 * 
 * @author acgist
 * 
 * @version 1.0.0
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

/**
 * 消息类型
 */
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

extern thread_local lifuren::message::Type thread_message_type; // 当前线程消息类型

/**
 * 注册消息通知
 */
extern void registerMessageCallback(
    Type type, // 消息类型
    std::function<void(bool, const char*)> callback // 通知回调
);

/**
 * 取消消息通知
 */
extern void unregisterMessageCallback(
    Type type // 消息类型
);

/**
 * 发送消息通知
 */
extern void sendMessage(
    const char* message, // 消息内容
    bool finish = false  // 是否完成
);

/**
 * 发送消息通知
 */
extern void sendMessage(
    Type type, // 消息类型
    const char* message, // 消息内容
    bool finish = false  // 是否完成
);

} // END OF lifuren::message

#endif // END OF LFR_HEADER_BOOT_MESSAGE_HPP
