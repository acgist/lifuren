#include "lifuren/PoetizeClient.hpp"

lifuren::PoetizeClient::PoetizeClient() {
}

lifuren::PoetizeClient::~PoetizeClient() {
}

std::unique_ptr<lifuren::PoetizeClient> lifuren::PoetizeClient::getClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_POETIZE_SHIFO_RNN) {
        return std::make_unique<lifuren::ShifoPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SHIMO_RNN) {
        return std::make_unique<lifuren::ShimoPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SHIGUI_RNN) {
        return std::make_unique<lifuren::ShiguiPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SHIXIAN_RNN) {
        return std::make_unique<lifuren::ShixianPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SHISHENG_RNN) {
        return std::make_unique<lifuren::ShishengPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_LIDU_RNN) {
        return std::make_unique<lifuren::LiduPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_SUXIN_RNN) {
        return std::make_unique<lifuren::SuxinPoetizeClient>();
    } else if(client == lifuren::config::CONFIG_POETIZE_WANYUE_RNN) {
        return std::make_unique<lifuren::WanyuePoetizeClient>();
    } else {
        return nullptr;
    }
}
