/**
 * 模型
 * 
 * @author acgist
 */
#pragma once

#include "torch/torch.h"

#include "./config/Setting.hpp"

namespace lifuren {

/**
 * 李夫人模型
 * 
 * @param M 模型配置
 * 
 * @author acgist
 */
template <typename M>
class LFRModel {

static_assert(std::is_base_of_v<lifuren::ModelSetting, M>, "必须继承模型配置");

public:
    // 基本配置
    lifuren::Setting setting;
    // 模型配置
    M modelSetting;

public:
    LFRModel();
    virtual ~LFRModel();
    /**
     * @param setting      基本配置
     * @param modelSetting 模型配置
     */
    LFRModel(const lifuren::Setting& setting, const M& modelSetting);

public:
    /**
     * 保存模型
     */
    virtual void save();
    /**
     * 加载模型
     */
    virtual void load();
    /**
     * 训练模型
     */
    virtual void train() = 0;
    /**
     * 测试模型
     */
    virtual void test() = 0;
    /**
     * 模型预测
     */
    virtual void pred() = 0;

};

}
