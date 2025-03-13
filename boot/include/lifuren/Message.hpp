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

namespace lifuren::message {

/**
 * 注册消息通知
 */
extern void registerMessageCallback(
    std::function<void(bool, const char*)> callback // 通知回调
);

/**
 * 取消消息通知
 */
extern void unregisterMessageCallback();

/**
 * 发送消息通知
 */
extern void sendMessage(
    const char* message, // 消息内容
    bool finish = false  // 是否完成
);

} // END OF lifuren::message

#endif // END OF LFR_HEADER_BOOT_MESSAGE_HPP
