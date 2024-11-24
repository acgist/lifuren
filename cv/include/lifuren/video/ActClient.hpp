/**
 * 导演终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_ACT_CLIENT_HPP
#define LFR_HEADER_CV_ACT_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

struct ActOptions {

    std::string model;
    std::string video;
    std::string output;
    
};

using ActModelClient = ModelClient<ActOptions, std::string>;

template<typename M>
using ActModelImplClient = ModelImplClient<ActOptions, std::string, M>;

extern std::unique_ptr<lifuren::ActModelClient> getActClient(const std::string& client);


/**
 * 导演终端
 */
template<typename M>
class ActClient : public StatefulClient, public ActModelImplClient<M> {

public:
    ActClient();
    virtual ~ActClient();

public:
    std::string pred(const ActOptions& input) override;
    void        pred(const ActOptions& input, ActModelClient::Callback callback) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_ACT_CLIENT_HPP
