#include "../../header/AudioGC.hpp"

lifuren::AudioGCModel::AudioGCModel() {
}

lifuren::AudioGCModel::~AudioGCModel() {
}

lifuren::AudioGCModel::AudioGCModel(lifuren::Setting& setting) : LFRModel(setting) {
};

void lifuren::AudioGCModel::predict(lifuren::AudioGCPredictSetting& setting) {
};

void lifuren::AudioGCModel::training(lifuren::AudioGCTrainingSetting& setting) {
};
