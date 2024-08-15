#include "lifuren/Client.hpp"

void lifuren::ChatClient::appendMessage(const lifuren::chat::Role& role, const std::string& message, bool done) {
    if(this->historyMessages.empty()) {
        this->historyMessages.push_back({
            .role    = role,
            .message = message,
            .done    = done
        });
    } else {
        lifuren::chat::ChatMessage& oldMessage = this->historyMessages.back();
        if(oldMessage.done) {
            this->historyMessages.push_back({
                .role    = role,
                .message = message,
                .done    = done
            });
        } else {
            oldMessage.message = oldMessage.message + message;
            oldMessage.done    = done;
        }
    }
}

void lifuren::ChatClient::reset() {
    this->historyMessages.clear();
}
