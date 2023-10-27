/**
 * 通用模型
 * 
 * @author acgist
 */
#pragma once

#include "Setting.hpp"

namespace lifuren {

/**
 * 李夫人模型
 * 
 * @param P 预测配置
 * @param T 训练配置
 * 
 * @author acgist
 */
template <class P, class T>
class LFRModel {

static_assert(std::is_base_of_v<lifuren::PredictSetting,  P>, "必须继承训练配置");
static_assert(std::is_base_of_v<lifuren::TrainingSetting, T>, "必须继承预测配置");

public:
    /**
     * 基本配置
     */
    lifuren::Setting setting;

public:
    LFRModel() {};
    virtual ~LFRModel() {};
    /**
     * @param setting 基本配置
     */
    LFRModel(lifuren::Setting& setting) : setting(setting) {};
    /**
     * 保存模型
     */
    virtual void save();
    /**
     * 加载模型
     */
    virtual void load();
    /**
     * 模型预测
     * 
     * @param 预测设置
     */
    virtual void predict(P& setting) = 0;
    /**
     * 训练模型
     * 
     * @param setting 训练设置
     */
    virtual void training(T& setting) = 0;

};

}