/**
 * 指针工具
 * 
 * @author acgist
 */
#pragma once

namespace {

// 删除指针
#ifndef DELETE_PTR
# define DELETE_PTR(ptr)       \
    if(this->ptr != nullptr) { \
        delete this->ptr;      \
        this->ptr = nullptr;   \
    }
#endif

}
