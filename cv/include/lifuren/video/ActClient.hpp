/**
 * 导演终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_ACT_CLIENT_HPP
#define LFR_HEADER_CV_ACT_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

struct ActParams {

    std::string model;  // 模型
    std::string video;  // 视频文件
    std::string output; // 输出文件
    
};

using ActModelClient = ModelClient<lifuren::config::ModelParams, ActParams, std::string>;

template<typename M>
using ActModelImplClient = ModelImplClient<lifuren::config::ModelParams, ActParams, std::string, M>;

extern std::unique_ptr<lifuren::ActModelClient> getActClient(const std::string& client);

/**
 * 导演终端
 */
template<typename M>
class ActClient : public ActModelImplClient<M> {

public:
    ActClient();
    virtual ~ActClient();

public:
    std::string pred(const ActParams& input) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_ACT_CLIENT_HPP
