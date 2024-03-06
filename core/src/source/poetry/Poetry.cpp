#include "../../header/Poetry.hpp"

lifuren::LabelText* lifuren::poetry::matchRule(const Poetry& poetry) {
    if(poetry.empty()) {
        return nullptr;
    }
    lifuren::LABEL_POETRY;
    return nullptr;
}

void lifuren::Poetry::participleSegment() {
    this->rule = lifuren::poetry::matchRule(&this);
}
