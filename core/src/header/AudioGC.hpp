/**
 * 音频生成
 * 
 * @author acgist
 */
#pragma once

#include "./LFRModel.hpp"

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
    void lifuren::AudioGCModel::predict(lifuren::AudioGCPredictSetting& setting) override;
    void lifuren::AudioGCModel::training(lifuren::AudioGCTrainingSetting& setting) override;

};

}