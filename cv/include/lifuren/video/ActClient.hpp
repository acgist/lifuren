/**
 * 导演终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_ACT_CLIENT_HPP
#define LFR_HEADER_CV_ACT_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

struct ActConfigOptions {

    std::string model;         // 模型
    std::string train_path {}; // 训练数据集路径
    std::string val_path   {}; // 验证数据集路径
    std::string test_path  {}; // 测试数据集路径

};

struct ActOptions {

    std::string video;  // 视频文件
    std::string output; // 输出文件
    
};

using ActModelClient = ModelClient<ActConfigOptions, ActOptions, std::string>;

template<typename M>
using ActModelImplClient = ModelImplClient<ActConfigOptions, ActOptions, std::string, M>;

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
    std::string pred(const ActOptions& input) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_ACT_CLIENT_HPP
