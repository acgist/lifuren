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
class PoetizeClient : public Client {

public:
    PoetizeClient();
    virtual ~PoetizeClient();

public:
    static std::unique_ptr<lifuren::PoetizeClient> getClient(const std::string& client);

};

class RNNPoetizeClient : public PoetizeClient {

public:
    RNNPoetizeClient();
    virtual ~RNNPoetizeClient();

};

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_POETIZE_CLIENT_HPP
