#include "lifuren/PoetizeClient.hpp"

lifuren::PoetizeClient::PoetizeClient() {
}

lifuren::PoetizeClient::~PoetizeClient() {
}

bool lifuren::PoetizeClient::stop() {
    this->changeState(false);
}

std::unique_ptr<lifuren::PoetizeClient> lifuren::PoetizeClient::getClient(const std::string& client) {
    if(client == "poetize-shifo-rnn") {
        return std::make_unique<lifuren::ShifoRNNPoetizeClient>();
    } else if(client == "poetize-shimo-rnn") {
        return std::make_unique<lifuren::ShimoRNNPoetizeClient>();
    } else if(client == "poetize-shigui-rnn") {
        return std::make_unique<lifuren::ShiguiRNNPoetizeClient>();
    } else if(client == "poetize-shixian-rnn") {
        return std::make_unique<lifuren::ShixianRNNPoetizeClient>();
    } else if(client == "poetize-shisheng-rnn") {
        return std::make_unique<lifuren::ShishengRNNPoetizeClient>();
    } else if(client == "poetize-lidu-rnn") {
        return std::make_unique<lifuren::LiduRNNPoetizeClient>();
    } else if(client == "poetize-suxin-rnn") {
        return std::make_unique<lifuren::SuxinRNNPoetizeClient>();
    } else if(client == "poetize-wanyue-rnn") {
        return std::make_unique<lifuren::WanyueRNNPoetizeClient>();
    } else {
        return nullptr;
    }
}
