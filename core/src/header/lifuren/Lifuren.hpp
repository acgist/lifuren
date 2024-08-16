/**
 * 李夫人
 */
#ifndef LFR_HEADER_CORE_LIFUREN_HPP
#define LFR_HEADER_CORE_LIFUREN_HPP

#include <cstddef>

namespace lifuren {

/**
 * 加载所有全局配置
 */
extern void loadConfig() noexcept;

/**
 * @return ID(yyyyMMddHHmmssxxxx)
 */
extern size_t uuid() noexcept;

}

#endif // LFR_HEADER_CORE_LIFUREN_HPP
