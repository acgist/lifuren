/**
 * RAII工具
 * 
 * @author acgist
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

namespace lifuren {

class Finally {

private:
    std::function<void(void)> finally{ nullptr };

public:
    Finally(Finally&)  = delete;
    Finally(Finally&&) = delete;
    Finally(std::function<void(void)> finally) : finally(finally) {
    }
    ~Finally() {
        finally();
    }

};

} // END lifuren

#endif // LFR_HEADER_CORE_RAII_HPP
