/**
 * 作曲终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_COMPOSE_CLIENT_HPP
#define LFR_HEADER_CV_COMPOSE_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

struct ComposeOptions {

    std::string model;
    std::string audio;
    std::string output;
    
};

using ComposeModelClient = ModelClient<ComposeOptions, std::string>;

template<typename M>
using ComposeModelImplClient = ModelImplClient<ComposeOptions, std::string, M>;

extern std::unique_ptr<lifuren::ComposeModelClient> getComposeClient(const std::string& client);

/**
 * 作曲终端
 */
template<typename M>
class ComposeClient : public ComposeModelImplClient<M> {

public:
    ComposeClient();
    virtual ~ComposeClient();

public:
    std::string pred(const ComposeOptions& input) override;
    void        pred(const ComposeOptions& input, ComposeModelClient::Callback callback) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_COMPOSE_CLIENT_HPP
