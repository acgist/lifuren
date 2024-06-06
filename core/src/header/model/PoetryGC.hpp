/**
 * 诗词内容生成
 * 
 * CycleGAN
 * Transformer
 * 
 * @author acgist
 */
#pragma once

#include <string>

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
class PoetryGCModel : public Model<PoetryGCModelConfig, std::string, std::string> {

public:
    PoetryGCModel();
    virtual ~PoetryGCModel();
    /**
     * @param config      基本配置
     * @param modelConfig 模型配置
     */
    PoetryGCModel(const lifuren::Config& config, const lifuren::PoetryGCModelConfig& modelConfig);

public:
    void train() override;
    void val() override;
    void test() override;
    std::string pred(std::string path) override;

};

}
