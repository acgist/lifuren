#include "lifuren/Client.hpp"

void lifuren::ChatClient::appendMessage(
    const lifuren::chat::Role& role,
    const std::string& message,
    const std::vector<std::string>& library,
    bool done
) {
    if(this->historyMessages.empty()) {
        this->historyMessages.push_back({
            .role    = role,
            .message = message,
            .library = library,
            .done    = done
        });
    } else {
        lifuren::chat::ChatMessage& oldMessage = this->historyMessages.back();
        if(oldMessage.done) {
            this->historyMessages.push_back({
                .role    = role,
                .message = message,
                .library = library,
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
