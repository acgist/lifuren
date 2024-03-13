#include "../header/Model.hpp"

template<typename M>
lifuren::LFRModel<M>::LFRModel() {
}

template<typename M>
lifuren::LFRModel<M>::~LFRModel() {
}

template<typename M>
lifuren::LFRModel<M>::LFRModel(const lifuren::Setting& setting, const M& modelSetting) : setting(setting), modelSetting(modelSetting) {
}

template<typename M>
void lifuren::LFRModel<M>::save() {
}

template<typename M>
void lifuren::LFRModel<M>::load() {
}
