/**
 * 指针工具
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_PTR_HPP
#define LFR_HEADER_CORE_PTR_HPP

// 删除指针
#ifndef LFR_DELETE_PTR
#define LFR_DELETE_PTR(ptr) \
    if(ptr != nullptr) {    \
        delete ptr;         \
        ptr = nullptr;      \
    }
#endif

// 删除指针
#ifndef LFR_DELETE_THIS_PTR
#define LFR_DELETE_THIS_PTR(ptr) \
    if(this->ptr != nullptr) {   \
        delete this->ptr;        \
        this->ptr = nullptr;     \
    }
#endif

namespace lifuren {

} // END lifuren

#endif // LFR_HEADER_CORE_PTR_HPP
