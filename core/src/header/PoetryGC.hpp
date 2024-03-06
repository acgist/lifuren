/**
 * 诗词内容生成
 * 
 * @author acgist
 */
#pragma once

#include "Model.hpp"
#include "Poetry.hpp"

namespace lifuren {

/**
 * 音频生成模型设置
 */
class AudioGCModelSetting : public lifuren::ModelSetting {
};

/**
 * 音频生成模型
 */
class AudioGCModel : public LFRModel<AudioGCModelSetting> {

public:
    /**
     * @param setting 基本配置
     */
    AudioGCModel(const lifuren::Setting& setting);
    virtual ~AudioGCModel();

public:
    /**
     * @param setting 模型配置
     */
    void train(const lifuren::AudioGCModelSetting& setting) override;
    /**
     * @param setting 模型配置
     */
    void predict(const lifuren::AudioGCModelSetting& setting) override;

};

}
