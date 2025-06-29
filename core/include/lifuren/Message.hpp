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
#ifndef LFR_HEADER_CORE_MESSAGE_HPP
#define LFR_HEADER_CORE_MESSAGE_HPP

#include <functional>

namespace lifuren::message {

/**
 * 注册消息通知回调
 * 
 * @param callback 消息通知回调
 */
extern void register_message_callback(std::function<void(const char*)> callback);

/**
 * 取消消息通知回调
 */
extern void unregister_message_callback();

/**
 * 发送消息通知
 * 
 * @param message 消息通知内容
 */
extern void sendMessage(const char* message);

} // END OF lifuren::message

#endif // END OF LFR_HEADER_CORE_MESSAGE_HPP
