#include "lifuren/video/ActClient.hpp"

#include "lifuren/video/ActModel.hpp"

template<typename M>
lifuren::ActClient<M>::ActClient() {
}

template<typename M>
lifuren::ActClient<M>::~ActClient() {
}

template<typename M>
std::string lifuren::ActClient<M>::pred(const ActParams& input) {
    // TODO: 实现
    return {};
};

std::unique_ptr<lifuren::ActModelClient> lifuren::getActClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_ACT_TANGXIANZU) {
        // return std::make_unique<lifuren::ActModelClient<TanxianzuModel>>();
    } else if(client == lifuren::config::CONFIG_ACT_GUANHANQING) {
        // return std::make_unique<lifuren::ActModelClient<GuanhanqingModel>>();
    } else {
        return nullptr;
    }
    return nullptr;
}
