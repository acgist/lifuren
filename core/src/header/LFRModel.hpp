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
 * @param T 训练配置
 * @param P 预测配置
 * 
 * @author acgist
 */
template <class T, class P>
class LFRModel {

static_assert(std::is_base_of<lifuren::PredictSetting,  T>, "必须继承训练配置");
static_assert(std::is_base_of<lifuren::TrainingSetting, P>, "必须继承预测配置");

public:
    /**
     * 基本配置
     */
    lifuren::Setting setting;

public:
    LFRModel();
    virtual ~LFRModel();
    /**
     * @param setting 基本配置
     */
    LFRModel(lifuren::Setting setting) : setting(setting);
    /**
     * 预测
     * 
     * @param 预测设置
     */
    void predict(T setting);
    /**
     * 训练
     * 
     * @param setting 训练设置
     */
    void training(P setting);

};

}