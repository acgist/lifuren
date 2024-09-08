#include "lifuren/PoetizeClient.hpp"

lifuren::PoetizeClient::PoetizeClient() {
}

lifuren::PoetizeClient::~PoetizeClient() {
}

std::unique_ptr<lifuren::PoetizeClient> lifuren::PoetizeClient::getClient(const std::string& client) {
    if(client == "poetize-rnn") {
        return std::make_unique<lifuren::RNNPoetizeClient>();
    } else {
        return nullptr;
    }
}
