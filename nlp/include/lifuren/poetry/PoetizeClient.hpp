/**
 * 诗词终端
 * 
 * 李杜
 * 诗仙
 * 诗圣
 * 诗佛
 * 诗鬼
 * 诗魔
 * 苏辛
 * 婉约
 * 
 * 模型实现：李杜、苏辛
 * 
 * RNN/GRU/LSTM/诗词填空/自监督学习
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_NLP_POETIZE_CLIENT_HPP
#define LFR_HEADER_NLP_POETIZE_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

struct PoetizeOptions {

    std::string model;  // 模型
    std::string rhythm; // 格律
    std::vector<std::string> prompts; // 提示

};

using PoetizeModelClient = ModelClient<PoetizeOptions, std::string>;

template<typename M>
using PoetizeModelImplClient = ModelImplClient<PoetizeOptions, std::string, M>;

extern std::unique_ptr<lifuren::PoetizeModelClient> getPoetizeClient(const std::string& client);

/**
 * 诗词终端
 */
template<typename M>
class PoetizeClient : public StatefulClient, public PoetizeModelImplClient<M> {

public:
    PoetizeClient();
    virtual ~PoetizeClient();

public:
    std::string pred(const PoetizeOptions& input) override;
    void        pred(const PoetizeOptions& input, PoetizeModelClient::Callback callback) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_POETIZE_CLIENT_HPP
