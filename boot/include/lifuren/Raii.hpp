/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * RAII
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_BOOT_RAII_HPP
#define LFR_HEADER_BOOT_RAII_HPP

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
    std::function<void(void)> finally{ nullptr }; // 资源释放

public:
    Finally() = delete;
    Finally(const Finally& ) = delete;
    Finally(      Finally&&) = delete;
    Finally& operator=(const Finally& ) = delete;
    Finally& operator=(      Finally&&) = delete;
    /**
     * @param finally 资源释放
     */
    Finally(
        std::function<void(void)> finally
    ) : finally(finally) {
    }
    ~Finally() {
        this->finally();
    }

};

} // END lifuren

#endif // LFR_HEADER_BOOT_RAII_HPP
