/**
 * 音频生成
 * 根据关键字生成音频数据
 * 
 * @author acgist
 */
#pragma once

#include "./Model.hpp"

namespace lifuren {

class AudioGCPredictSetting : public lifuren::PredictSetting {
};

class AudioGCTrainingSetting : public lifuren::TrainingSetting {
};

class AudioGCModel : public LFRModel<AudioGCPredictSetting, AudioGCTrainingSetting> {

public:
    AudioGCModel() {};
    ~AudioGCModel() {};
    /**
     * @param setting 基本配置
     */
    AudioGCModel(lifuren::Setting& setting) : LFRModel(setting) {};

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
