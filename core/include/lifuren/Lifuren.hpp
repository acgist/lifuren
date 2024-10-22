/**
 * 李夫人
 */
#ifndef LFR_HEADER_CORE_LIFUREN_HPP
#define LFR_HEADER_CORE_LIFUREN_HPP

#include <cstdlib>
#include <cstdint>

namespace lifuren {

/**
 * 注意：一秒钟的并发不能超过一万
 * 
 * @return ID(yyyyMMddHHmmssxxxx)
 */
extern size_t uuid() noexcept;

}

#endif // LFR_HEADER_CORE_LIFUREN_HPP
