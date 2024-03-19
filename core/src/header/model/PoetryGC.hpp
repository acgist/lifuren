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
class PoetryGCModelSetting : public lifuren::ModelSetting {
};

/**
 * 诗词生成模型
 */
class PoetryGCModel : public LFRModel<PoetryGCModelSetting> {

public:
    /**
     * @param setting        基本配置
     * @param modelSetting 模型配置
     */
    PoetryGCModel(const lifuren::Setting& setting, const lifuren::PoetryGCModelSetting& modelSetting);
    virtual ~PoetryGCModel();

public:
    void train() override;
    void val() override;
    void test() override;
    void pred() override;

};

}
