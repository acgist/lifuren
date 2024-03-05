/**
 * 模型
 * 
 * @author acgist
 */
#pragma once

#include "Setting.hpp"

#include "torch/torch.h"

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

public:
    LFRModel();
    virtual ~LFRModel();
    /**
     * @param setting 基本配置
     */
    LFRModel(const lifuren::Setting& setting);

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
     * 
     * @param setting 模型设置
     */
    virtual void train(const M& setting) = 0;
    /**
     * 模型预测
     * 
     * @param 模型设置
     */
    virtual void predict(const M& setting) = 0;

};

}
