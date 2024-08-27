/**
 * 模型
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_MODEL_MODEL_HPP
#define LFR_HEADER_MODEL_MODEL_HPP

#include "lifuren/config/Config.hpp"

namespace lifuren {

/**
 * 李夫人模型
 * 
 * @param M 模型配置
 * @param I 预测输入
 * @param R 预测输出
 * 
 * @author acgist
 */
template<typename M, typename I, typename R>
// TODO requires std::derived_from<M, lifuren::config::ModelConfig>
class Model {

// TODO: concept derived_from
static_assert(std::is_base_of_v<lifuren::config::ModelConfig, M>, "必须继承模型配置");

public:
    // 模型配置
    M modelConfig;
    // TODO: 训练数据集、验证数据集、测试数据集

public:
    Model();
    virtual ~Model();
    /**
     * @param config      基本配置
     * @param modelConfig 模型配置
     */
    Model(const M& modelConfig);

public:
    // 保存模型
    virtual void save();
    // 加载模型
    virtual void load();
    // 训练模型
    virtual void train() = 0;
    // 验证模型
    virtual void val() = 0;
    // 测试模型
    virtual void test() = 0;
    // 模型微调
    virtual void finetune() = 0;
    // 模型量化
    virtual void quantization() = 0;
    /**
     * 模型预测
     * 
     * @param input 预测输入
     * 
     * @return 预测输出
     */
    virtual R pred(I input) = 0;
    // 训练验证
    virtual void trainAndVal();

};

// TODO: concept

template<typename M, typename I, typename R>
lifuren::Model<M, I, R>::Model() {
}

template<typename M, typename I, typename R>
lifuren::Model<M, I, R>::~Model() {
}

template<typename M, typename I, typename R>
lifuren::Model<M, I, R>::Model(const M& modelConfig) : modelConfig(modelConfig) {
}

template<typename M, typename I, typename R>
void lifuren::Model<M, I, R>::save() {
}

template<typename M, typename I, typename R>
void lifuren::Model<M, I, R>::load() {
}

template<typename M, typename I, typename R>
void lifuren::Model<M, I, R>::trainAndVal() {
}

}

#endif // LFR_HEADER_MODEL_MODEL_HPP
