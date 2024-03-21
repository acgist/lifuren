#include "../header/Model.hpp"

template<typename M>
lifuren::Model<M>::Model() {
}

template<typename M>
lifuren::Model<M>::~Model() {
}

template<typename M>
lifuren::Model<M>::Model(const lifuren::Config& config, const M& modelConfig) : config(config), modelConfig(modelConfig) {
}

template<typename M>
void lifuren::Model<M>::save() {
}

template<typename M>
void lifuren::Model<M>::load() {
}
