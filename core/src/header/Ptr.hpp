/**
 * 指针工具
 * 
 * @author acgist
 */
#pragma once

namespace {

// 删除指针
#ifndef LFR_DELETE_PTR
# define LFR_DELETE_PTR(ptr)   \
    if(this->ptr != nullptr) { \
        delete this->ptr;      \
        this->ptr = nullptr;   \
    }
#endif

}
