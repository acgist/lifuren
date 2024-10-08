#include "lifuren/Client.hpp"

lifuren::StatefulClient::StatefulClient() {
}

lifuren::StatefulClient::~StatefulClient() {
}

const bool& lifuren::StatefulClient::isRunning() const {
    return this->running;
}

void lifuren::StatefulClient::changeState() {
    std::unique_lock<std::mutex> lock(this->mutex);
    this->running = !this->running;
}

void lifuren::StatefulClient::changeState(bool running) {
    std::unique_lock<std::mutex> lock(this->mutex);
    this->running = running;
}

bool lifuren::StatefulClient::stop() {
    this->changeState(false);
    return true;
}
