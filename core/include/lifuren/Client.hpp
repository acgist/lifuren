/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 终端
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
class ModelClient {

public:
    virtual bool save(const std::string& path = "./lifuren.pt") = 0;
    virtual bool load(const std::string& path = "./lifuren.pt", C params = {}) = 0;
    virtual void trainValAndTest(C params = {}, const bool val = true, const bool test = true) = 0;
    virtual std::tuple<bool, O> pred(const I& input) = 0;

};

/**
 * 模型终端
 * 
 * @param C 模型配置
 * @param I 模型输入
 * @param O 模型输出
 * @param M 模型实现
 */
template<typename C, typename I, typename O, typename M>
class ModelClientImpl : public ModelClient<C, I, O> {

protected:
    std::unique_ptr<M> model{ nullptr }; // 模型实现

public:
    virtual bool save(const std::string& path = "./lifuren.pt") override;
    virtual bool load(const std::string& path = "./lifuren.pt", C params = {}) override;
    virtual void trainValAndTest(C params = {}, const bool val = true, const bool test = true) override;
    virtual std::tuple<bool, O> pred(const I& input) = 0;

};

} // END OF lifuren

template<typename C, typename I, typename O, typename M>
bool lifuren::ModelClientImpl<C, I, O, M>::save(const std::string& path) {
    if(!this->model) {
        return false;
    }
    return this->model->save(path);
}

template<typename C, typename I, typename O, typename M>
bool lifuren::ModelClientImpl<C, I, O, M>::load(const std::string& path, C params) {
    if(this->model) {
        return true;
    } else {
        this->model = std::make_unique<M>(params);
    }
    return this->model->load(path);
}

template<typename C, typename I, typename O, typename M>
void lifuren::ModelClientImpl<C, I, O, M>::trainValAndTest(C params, const bool val, const bool test) {
    if(!this->model) {
        this->model = std::make_unique<M>(params);
    }
    this->model->trainValAndTest(val, test);
}

#endif // END OF LFR_HEADER_CORE_CLIENT_HPP
