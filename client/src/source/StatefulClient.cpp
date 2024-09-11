#include "lifuren/Client.hpp"

lifuren::StatefulClient::StatefulClient() {
}

lifuren::StatefulClient::~StatefulClient() {
}

const bool& lifuren::StatefulClient::isRunning() const {
    return this->running;
}

void lifuren::StatefulClient::changeState() {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->running = !this->running;
}

void lifuren::StatefulClient::changeState(bool running) {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->running = running;
}
