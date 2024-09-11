/**
 * 诗词终端
 * 
 * 自监督学习
 * 
 * RNN/GRU/LSTM/诗词填空
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

class ShifoRNNPoetizeClient : public PoetizeClient {

public:
    ShifoRNNPoetizeClient();
    virtual ~ShifoRNNPoetizeClient();

};

class ShimoRNNPoetizeClient : public PoetizeClient {

public:
    ShimoRNNPoetizeClient();
    virtual ~ShimoRNNPoetizeClient();

};

class ShiguiRNNPoetizeClient : public PoetizeClient {

public:
    ShiguiRNNPoetizeClient();
    virtual ~ShiguiRNNPoetizeClient();

};

class ShixianRNNPoetizeClient : public PoetizeClient {

public:
    ShixianRNNPoetizeClient();
    virtual ~ShixianRNNPoetizeClient();

};

class ShishengRNNPoetizeClient : public PoetizeClient {

public:
    ShishengRNNPoetizeClient();
    virtual ~ShishengRNNPoetizeClient();

};

class LiduRNNPoetizeClient : public PoetizeClient {

public:
    LiduRNNPoetizeClient();
    virtual ~LiduRNNPoetizeClient();

};

class SuxinRNNPoetizeClient : public PoetizeClient {

public:
    SuxinRNNPoetizeClient();
    virtual ~SuxinRNNPoetizeClient();

};

class WanyueRNNPoetizeClient : public PoetizeClient {

public:
    WanyueRNNPoetizeClient();
    virtual ~WanyueRNNPoetizeClient();

};

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_POETIZE_CLIENT_HPP
