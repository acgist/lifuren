/**
 * 诗词内容生成
 * 
 * CycleGAN
 * Transformer
 * 
 * @author acgist
 */
#pragma once

#include "../Model.hpp"
#include "./Poetry.hpp"

namespace lifuren {

/**
 * 诗词生成模型设置
 */
class PoetryGCModelConfig : public lifuren::ModelConfig {
};

/**
 * 诗词生成模型
 */
class PoetryGCModel : public Model<PoetryGCModelConfig> {

public:
    /**
     * @param config      基本配置
     * @param modelConfig 模型配置
     */
    PoetryGCModel(const lifuren::Config& config, const lifuren::PoetryGCModelConfig& modelConfig);
    virtual ~PoetryGCModel();

public:
    void train() override;
    void val() override;
    void test() override;
    void pred() override;

};

}
