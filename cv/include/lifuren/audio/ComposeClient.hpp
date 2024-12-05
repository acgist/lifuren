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
 * 作曲训练配置
 */
struct ComposeConfigOptions {

    std::string model      {}; // 模型路径
    std::string train_path {}; // 训练数据集路径
    std::string val_path   {}; // 验证数据集路径
    std::string test_path  {}; // 测试数据集路径

};

/**
 * 作曲推理配置
 */
struct ComposeOptions {

    std::string audio;  // 音频文件
    std::string output; // 输出位置
    
};

using ComposeModelClient = ModelClient<ComposeConfigOptions, ComposeOptions, std::string>;

template<typename M>
using ComposeModelImplClient = ModelImplClient<ComposeConfigOptions, ComposeOptions, std::string, M>;

extern std::unique_ptr<lifuren::ComposeModelClient> getComposeClient(const std::string& client);

/**
 * 作曲终端
 */
template<typename M>
class ComposeClient : public ComposeModelImplClient<M> {

public:
    ComposeClient(ComposeConfigOptions config);
    virtual ~ComposeClient();

public:
    std::string pred(const ComposeOptions& input) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_COMPOSE_CLIENT_HPP
