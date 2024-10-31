#include "lifuren/ComposeClient.hpp"

lifuren::ComposeClient::ComposeClient(lifuren::ComposeClient::ComposeCallback callback) : callback(callback) {
}

lifuren::ComposeClient::~ComposeClient() {
}

std::unique_ptr<lifuren::ComposeClient> lifuren::ComposeClient::getClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_COMPOSE_SHIKUANG) {
        return std::make_unique<lifuren::ShikuangComposeClient>();
    } else if(client == lifuren::config::CONFIG_COMPOSE_LIGUINIAN) {
        return std::make_unique<lifuren::LiguinianComposeClient>();
    } else {
        return nullptr;
    }
}
