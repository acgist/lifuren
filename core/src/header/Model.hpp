/**
 * 模型
 * 
 * @author acgist
 */
#pragma once

#include "torch/torch.h"
#include "torch/script.h"

#include "./config/Config.hpp"

namespace lifuren {

namespace model   {

extern void loadChatGLM();

extern void loadStableDiffusion();

}

/**
 * 李夫人模型
 * 
 * @param M 模型配置
 * 
 * @author acgist
 */
template <typename M>
class Model {

static_assert(std::is_base_of_v<lifuren::ModelConfig, M>, "必须继承模型配置");

public:
    // 基本配置
    lifuren::Config config;
    // 模型配置
    M modelConfig;

public:
    Model();
    virtual ~Model();
    /**
     * @param config      基本配置
     * @param modelConfig 模型配置
     */
    Model(const lifuren::Config& config, const M& modelConfig);

public:
    // 保存模型
    virtual void save();
    // 加载模型
    virtual void load();
    // 训练模型
    virtual void train() = 0;
    // 验证模型
    virtual void val()   = 0;
    // 测试模型
    virtual void test()  = 0;
    // 模型预测
    virtual void pred()  = 0;
    // 训练验证
    virtual void trainAndVal();

};

}
