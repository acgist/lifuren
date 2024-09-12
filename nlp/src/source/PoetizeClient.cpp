#include "lifuren/PoetizeClient.hpp"

lifuren::PoetizeClient::PoetizeClient() {
}

lifuren::PoetizeClient::~PoetizeClient() {
}

bool lifuren::PoetizeClient::stop() {
    this->changeState(false);
    return true;
}

std::unique_ptr<lifuren::PoetizeClient> lifuren::PoetizeClient::getClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_POETIZE_SHIFO_RNN) {
        return std::make_unique<lifuren::ShifoRNNPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SHIMO_RNN) {
        return std::make_unique<lifuren::ShimoRNNPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SHIGUI_RNN) {
        return std::make_unique<lifuren::ShiguiRNNPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SHIXIAN_RNN) {
        return std::make_unique<lifuren::ShixianRNNPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SHISHENG_RNN) {
        return std::make_unique<lifuren::ShishengRNNPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_LIDU_RNN) {
        return std::make_unique<lifuren::LiduRNNPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SUXIN_RNN) {
        return std::make_unique<lifuren::SuxinRNNPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_WANYUE_RNN) {
        return std::make_unique<lifuren::WanyueRNNPoetizeClient>();
    } else {
        return nullptr;
    }
}
