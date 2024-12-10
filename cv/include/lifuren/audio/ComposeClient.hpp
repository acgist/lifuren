/**
 * 作曲终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_COMPOSE_CLIENT_HPP
#define LFR_HEADER_CV_COMPOSE_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

/**
 * 作曲推理配置
 */
struct ComposeParams {

    std::string model;  // 模型路径
    std::string audio;  // 音频文件
    std::string output; // 输出位置
    
};

using ComposeModelClient = ModelClient<lifuren::config::ModelParams, ComposeParams, std::string>;

template<typename M>
using ComposeModelImplClient = ModelImplClient<lifuren::config::ModelParams, ComposeParams, std::string, M>;

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
    std::string pred(const ComposeParams& input) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_COMPOSE_CLIENT_HPP
