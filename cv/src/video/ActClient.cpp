#include "lifuren/video/ActClient.hpp"

lifuren::ActClient::ActClient(lifuren::ActClient::ActCallback callback) : callback(callback) {
}

lifuren::ActClient::~ActClient() {
}

std::unique_ptr<lifuren::ActClient> lifuren::ActClient::getClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_ACT_GUANHANQIN) {
        return std::make_unique<lifuren::GuanhanqinActClient>();
    } else if(client == lifuren::config::CONFIG_ACT_TANGXIANZU) {
        return std::make_unique<lifuren::TangxianzuActClient>();
    } else {
        return nullptr;
    }
}
