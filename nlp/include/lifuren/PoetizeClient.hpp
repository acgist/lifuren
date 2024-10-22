/**
 * 诗词终端
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
 * 诗佛终端
 */
class ShifoPoetizeClient : public PoetizeClient {

public:
    ShifoPoetizeClient();
    virtual ~ShifoPoetizeClient();

};

/**
 * 诗魔终端
 */
class ShimoPoetizeClient : public PoetizeClient {

public:
    ShimoPoetizeClient();
    virtual ~ShimoPoetizeClient();

};

/**
 * 诗鬼终端
 */
class ShiguiPoetizeClient : public PoetizeClient {

public:
    ShiguiPoetizeClient();
    virtual ~ShiguiPoetizeClient();

};

/**
 * 诗仙终端
 */
class ShixianPoetizeClient : public PoetizeClient {

public:
    ShixianPoetizeClient();
    virtual ~ShixianPoetizeClient();

};

/**
 * 诗圣终端
 */
class ShishengPoetizeClient : public PoetizeClient {

public:
    ShishengPoetizeClient();
    virtual ~ShishengPoetizeClient();

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

/**
 * 婉约终端
 */
class WanyuePoetizeClient : public PoetizeClient {

public:
    WanyuePoetizeClient();
    virtual ~WanyuePoetizeClient();

};

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_POETIZE_CLIENT_HPP
