#include "lifuren/audio/ComposeClient.hpp"

#include "lifuren/audio/ComposeModel.hpp"

template<typename M>
lifuren::ComposeClient<M>::ComposeClient(ComposeConfigOptions config) : ComposeModelImplClient<M>(std::move(config)) {
}

template<typename M>
lifuren::ComposeClient<M>::~ComposeClient() {
}

template<typename M>
std::string lifuren::ComposeClient<M>::pred(const ComposeOptions& input) {
    // TODO: 实现
    return {};
};

std::unique_ptr<lifuren::ComposeModelClient> lifuren::getComposeClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_COMPOSE_SHIKUANG) {
        // return std::make_unique<lifuren::ComposeClient<ShikuangModel>>();
    } else if(client == lifuren::config::CONFIG_COMPOSE_LIGUINIAN) {
        // return std::make_unique<lifuren::ComposeClient<LiguinianModel>>();
    } else {
        return nullptr;
    }
    return nullptr;
}
