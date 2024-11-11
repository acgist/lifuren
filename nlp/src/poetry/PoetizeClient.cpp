#include "lifuren/poetry/PoetizeClient.hpp"

lifuren::PoetizeClient::PoetizeClient() {
}

lifuren::PoetizeClient::~PoetizeClient() {
}

std::unique_ptr<lifuren::PoetizeClient> lifuren::PoetizeClient::getClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_POETIZE_LIDU) {
        return std::make_unique<lifuren::LiduPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SUXIN) {
        return std::make_unique<lifuren::SuxinPoetizeClient>();
    } else {
        return nullptr;
    }
}
