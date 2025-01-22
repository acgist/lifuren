/**
 * 诗词终端
 * 
 * 诗仙
 * 诗圣
 * 诗佛
 * 诗鬼
 * 诗魔
 * 李杜
 * 苏辛
 * 婉约
 * 
 * 模型实现：李杜、苏辛
 * 
 * RNN/GRU/LSTM/诗词填空/自监督学习
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_NLP_POETRY_HPP
#define LFR_HEADER_NLP_POETRY_HPP

#include "lifuren/Client.hpp"

namespace lifuren::poetry {


/**
 * 诗词推理配置
 */
struct PoetryParams {

    std::string model;                // 模型文件
    std::string rhythm;               // 格律
    std::vector<std::string> prompts; // 提示

};

using PoetryModelClient = ModelClient<lifuren::config::ModelParams, PoetryParams, std::string>;

template<typename M>
using PoetryModelImplClient = ModelImplClient<lifuren::config::ModelParams, PoetryParams, std::string, M>;

/**
 * 诗词终端
 */
template<typename M>
class PoetryClient : public PoetryModelImplClient<M> {

public:
    std::tuple<bool, std::string> pred(const PoetryParams& input) override;

};

template<typename M>
using PoetizeClient = PoetryClient<M>;

extern std::unique_ptr<lifuren::poetry::PoetryModelClient> getPoetryClient(const std::string& client);

} // END OF lifuren::poetry

#endif // LFR_HEADER_NLP_POETRY_HPP
