#include "lifuren/poetry/PoetizeClient.hpp"

#include "lifuren/poetry/PoetizeModel.hpp"

template<typename M>
lifuren::PoetizeClient<M>::PoetizeClient() {
}

template<typename M>
lifuren::PoetizeClient<M>::~PoetizeClient() {
}

template<typename M>
std::string lifuren::PoetizeClient<M>::pred(const PoetizeOptions& input) {
    // TODO: 实现
    return {};
};

std::unique_ptr<lifuren::PoetizeModelClient> lifuren::getPoetizeClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_POETIZE_LIDU) {
        // return std::make_unique<lifuren::PoetizeClient<LiduModel>>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SUXIN) {
        return std::make_unique<lifuren::PoetizeClient<SuxinModel>>();
    } else {
        return nullptr;
    }
    return nullptr;
}
