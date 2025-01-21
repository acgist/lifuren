/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * RAII工具
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_RAII_HPP
#define LFR_HEADER_CORE_RAII_HPP

#include <functional>

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

// 删除指针
#ifndef LFR_DELETE_ARRAY_PTR
#define LFR_DELETE_ARRAY_PTR(ptr) \
    if(ptr != nullptr) {          \
        delete[] ptr;             \
        ptr = nullptr;            \
    }
#endif

// 删除指针
#ifndef LFR_DELETE_THIS_ARRAY_PTR
#define LFR_DELETE_THIS_ARRAY_PTR(ptr) \
    if(this->ptr != nullptr) {         \
        delete[] this->ptr;            \
        this->ptr = nullptr;           \
    }
#endif

namespace lifuren {

/**
 * 自动释放
 * 
 * lifuren::Finally finally([]() {
 *     ...释放资源
 * });
 */
class Finally {

private:
    // 资源释放
    std::function<void(void)> finally{ nullptr };

public:
    Finally(const Finally& ) = delete;
    Finally(      Finally&&) = delete;
    Finally& operator=(const Finally& ) = delete;
    Finally& operator=(      Finally&&) = delete;
    Finally(
        std::function<void(void)> finally // 资源释放
    ) : finally(finally) {
    }
    ~Finally() {
        this->finally();
    }

};

} // END lifuren

#endif // LFR_HEADER_CORE_RAII_HPP
