#include "../../header/Poetry.hpp"

#include <algorithm>

#include "Strings.hpp"
#include "Collections.hpp"

std::string lifuren::poetry::beautify(const std::string& segment) {
    std::string ret;
    if(segment.empty()) {
        return ret;
    }
    std::vector<std::string> vector = lifuren::collections::split(segment, lifuren::poetry::POETRY_BEAUTIFY_DELIM, true);
    for(
        auto iterator = vector.begin();
        iterator != vector.end();
        ++iterator
    ) {
        if(iterator->empty()) {
            continue;
        }
        ret += lifuren::strings::trim(*iterator);
        ret += "\n";
    }
    return ret;
}

lifuren::Poetry& lifuren::Poetry::preproccess() {
    if(this->title.empty() && !this->rhythmic.empty()) {
        this->title = this->rhythmic;
    } else if(!this->title.empty() && this->rhythmic.empty()) {
        this->rhythmic = this->title;
    }
    if(
        this->segment.empty() &&
        this->simpleSegment.empty() &&
        !this->paragraphs.empty()
    ) {
        std::string segment;
        for(
            auto iterator = this->paragraphs.begin();
            iterator != this->paragraphs.end();
            ++iterator
        ) {
            if(iterator->empty()) {
                continue;
            }
            segment = lifuren::strings::trim(*iterator);
            this->segment += segment;
            lifuren::strings::replace(segment, lifuren::poetry::POETRY_SEGMENT_DELIM);
            this->simpleSegment += segment;
        }
    }
    return *this;
}

bool lifuren::Poetry::matchLabel() {
    std::vector<std::string> vector = lifuren::collections::split(segment, lifuren::poetry::POETRY_SEGMENT_DELIM);
    std::vector<uint32_t> segmentRule(vector.size());
    std::transform(vector.begin(), vector.end(), segmentRule.begin(), [](auto& v) -> uint32_t {
        return (uint32_t) lifuren::strings::length(v);
    });
    for(
        auto iterator = lifuren::LABEL_POETRY.begin();
        iterator != lifuren::LABEL_POETRY.end();
        ++iterator
    ) {
        LabelText& label = iterator->second;
        // TODO：验证词能否正确匹配或者添加词牌
        if(
            label.segmentSize == vector.size() &&
            label.segmentRule == segmentRule
        ) {
            this->label = &label;
            break;
        }
    }
    return this->label != nullptr;
}

bool lifuren::Poetry::participleSegment() {
    return true;
}
