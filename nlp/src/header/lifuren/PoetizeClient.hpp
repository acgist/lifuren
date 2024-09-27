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
    bool stop() override;

public:
    static std::unique_ptr<lifuren::PoetizeClient> getClient(const std::string& client);

};

/**
 * 诗佛终端
 */
class ShifoRNNPoetizeClient : public PoetizeClient {

public:
    ShifoRNNPoetizeClient();
    virtual ~ShifoRNNPoetizeClient();

};

/**
 * 诗魔终端
 */
class ShimoRNNPoetizeClient : public PoetizeClient {

public:
    ShimoRNNPoetizeClient();
    virtual ~ShimoRNNPoetizeClient();

};

/**
 * 诗鬼终端
 */
class ShiguiRNNPoetizeClient : public PoetizeClient {

public:
    ShiguiRNNPoetizeClient();
    virtual ~ShiguiRNNPoetizeClient();

};

/**
 * 诗仙终端
 */
class ShixianRNNPoetizeClient : public PoetizeClient {

public:
    ShixianRNNPoetizeClient();
    virtual ~ShixianRNNPoetizeClient();

};

/**
 * 诗圣终端
 */
class ShishengRNNPoetizeClient : public PoetizeClient {

public:
    ShishengRNNPoetizeClient();
    virtual ~ShishengRNNPoetizeClient();

};

/**
 * 李杜终端
 */
class LiduRNNPoetizeClient : public PoetizeClient {

public:
    LiduRNNPoetizeClient();
    virtual ~LiduRNNPoetizeClient();

};

/**
 * 苏辛终端
 */
class SuxinRNNPoetizeClient : public PoetizeClient {

public:
    SuxinRNNPoetizeClient();
    virtual ~SuxinRNNPoetizeClient();

};

/**
 * 婉约终端
 */
class WanyueRNNPoetizeClient : public PoetizeClient {

public:
    WanyueRNNPoetizeClient();
    virtual ~WanyueRNNPoetizeClient();

};

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_POETIZE_CLIENT_HPP
