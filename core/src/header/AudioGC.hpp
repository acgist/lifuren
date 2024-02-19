/**
 * 音频内容生成
 * 根据关键字生成音频数据
 * 
 * @author acgist
 */
#pragma once

#include "./Model.hpp"

namespace lifuren {

/**
 * 音频生成预测设置
 */
class AudioGCPredictSetting : public lifuren::PredictSetting {
};

/**
 * 音频生成训练设置
 */
class AudioGCTrainingSetting : public lifuren::TrainingSetting {
};

/**
 * 音频生成模型
 */
class AudioGCModel : public LFRModel<AudioGCPredictSetting, AudioGCTrainingSetting> {

public:
    AudioGCModel();
    ~AudioGCModel();
    /**
     * @param setting 基本配置
     */
    AudioGCModel(lifuren::Setting& setting);

public:
    /**
     * @param setting 预测配置
     */
    void predict(lifuren::AudioGCPredictSetting& setting) override;
    /**
     * @param setting 训练配置
     */
    void training(lifuren::AudioGCTrainingSetting& setting) override;

};

}
