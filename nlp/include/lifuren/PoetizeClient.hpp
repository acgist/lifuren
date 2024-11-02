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

/**
 * 诗词终端
 */
class PoetizeClient : public StatefulClient {

public:
    PoetizeClient();
    virtual ~PoetizeClient();

public:
    static std::unique_ptr<lifuren::PoetizeClient> getClient(const std::string& client);

};

/**
 * 李杜终端
 */
class LiduPoetizeClient : public PoetizeClient {

public:
    LiduPoetizeClient();
    virtual ~LiduPoetizeClient();

};

/**
 * 苏辛终端
 */
class SuxinPoetizeClient : public PoetizeClient {

public:
    SuxinPoetizeClient();
    virtual ~SuxinPoetizeClient();

};

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_POETIZE_CLIENT_HPP
