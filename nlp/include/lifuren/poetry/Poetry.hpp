/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 诗词
 * 
 * @author acgist
 * 
 * @version 1.0.0
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

extern bool datasetPepperPreprocessing(const std::string& path);

extern bool datasetPoetryPreprocessing(const std::string& path, const std::string& rag_type, const std::string& embedding_type);

} // END OF lifuren::poetry

#endif // END OF LFR_HEADER_NLP_POETRY_HPP
