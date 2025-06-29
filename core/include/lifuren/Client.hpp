/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 终端：提供训练、预测接口
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_CLIENT_HPP
#define LFR_HEADER_CORE_CLIENT_HPP

#include <tuple>
#include <memory>
#include <string>

#include "lifuren/Config.hpp"

namespace lifuren {

/**
 * 模型终端
 * 
 * @param C 模型配置
 * @param I 模型输入
 * @param O 模型输出
 */
template<typename C, typename I, typename O>
class Client {

public:
    virtual bool save(const std::string& path = "./lifuren.pt") = 0;
    virtual bool load(const std::string& path = "./lifuren.pt", C params = {}) = 0;
    virtual void trainValAndTest(C params = {}, const bool val = true, const bool test = true) = 0;
    virtual std::tuple<bool, O> pred(const I& input) = 0; // torch::NoGradGuard no_grad;

};

/**
 * 模型终端
 * 
 * @param C 模型配置
 * @param I 模型输入
 * @param O 模型输出
 * @param T 模型训练器
 */
template<typename C, typename I, typename O, typename T>
class ClientImpl : public Client<C, I, O> {

protected:
    std::unique_ptr<T> trainer{ nullptr }; // 模型训练器

public:
    virtual bool save(const std::string& path = "./lifuren.pt") override;
    virtual bool load(const std::string& path = "./lifuren.pt", C params = {}) override;
    virtual void trainValAndTest(C params = {}, const bool val = true, const bool test = true) override;
    virtual std::tuple<bool, O> pred(const I& input) = 0; // torch::NoGradGuard no_grad;

};

} // END OF lifuren

template<typename C, typename I, typename O, typename T>
bool lifuren::ClientImpl<C, I, O, T>::save(const std::string& path) {
    if(!this->trainer) {
        return false;
    }
    return this->trainer->save(path);
}

template<typename C, typename I, typename O, typename T>
bool lifuren::ClientImpl<C, I, O, T>::load(const std::string& path, C params) {
    if(this->trainer) {
        return true;
    } else {
        this->trainer = std::make_unique<T>(params);
    }
    return this->trainer->load(path);
}

template<typename C, typename I, typename O, typename T>
void lifuren::ClientImpl<C, I, O, T>::trainValAndTest(C params, const bool val, const bool test) {
    if(!this->trainer) {
        this->trainer = std::make_unique<T>(params);
        this->trainer->define();
    }
    this->trainer->trainValAndTest(val, test);
}

#endif // END OF LFR_HEADER_CORE_CLIENT_HPP
