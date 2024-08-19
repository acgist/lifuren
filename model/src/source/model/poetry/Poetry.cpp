#include "lifuren/model/Poetry.hpp"

#include <cstdint>
#include <algorithm>

#include "lifuren/Strings.hpp"
#include "lifuren/Collections.hpp"

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
    std::string content = lifuren::collections::join(this->paragraphs, "");
    this->paragraphs = lifuren::collections::split(
        content,
        lifuren::poetry::POETRY_BEAUTIFY_DELIM,
        true
    );
    this->simpleParagraphs = lifuren::collections::split(
        content,
        lifuren::poetry::POETRY_SEGMENT_DELIM
    );
    this->segment = lifuren::collections::join(this->paragraphs, "\n");
    this->simpleSegment = lifuren::collections::join(this->simpleParagraphs, "\n");
    return *this;
}

bool lifuren::Poetry::matchLabel() {
    std::vector<uint32_t> segmentRule(this->simpleParagraphs.size());
    std::transform(this->simpleParagraphs.begin(), this->simpleParagraphs.end(), segmentRule.begin(), [](auto& v) -> uint32_t {
        return (uint32_t) lifuren::strings::length(v);
    });
    for(
        auto iterator = lifuren::LABEL_POETRY.begin();
        iterator != lifuren::LABEL_POETRY.end();
        ++iterator
    ) {
        LabelText& labelRef = iterator->second;
        // TODO: 验证词能否正确匹配或者添加词牌
        if(
            labelRef.segmentSize == static_cast<int>(this->simpleParagraphs.size()) &&
            labelRef.segmentRule == segmentRule
        ) {
            this->label = &labelRef;
            break;
        }
    }
    return this->label != nullptr;
}

bool lifuren::Poetry::participle() {
    if(this->label == nullptr) {
        return false;
    }
    std::vector<uint32_t>& participleRuleRef = this->label->participleRule;
    std::string word;
    uint32_t pos = 0;
    auto paragraphsIterator = this->simpleParagraphs.begin();
    for(
        auto iterator = participleRuleRef.begin();
        iterator != participleRuleRef.end();
        ++iterator
    ) {
        word = lifuren::strings::substr(paragraphsIterator->c_str(), pos, *iterator);
        this->participleParagraphs.push_back(word);
        if(this->participleSegment.empty()) {
            this->participleSegment = word;
        } else {
            this->participleSegment = this->participleSegment + word;
        }
        if(pos >= paragraphsIterator->length()) {
            pos = 0;
            ++paragraphsIterator;
            if(paragraphsIterator == this->simpleParagraphs.end()) {
                break;
            }
            this->participleSegment += "\n";
        } else {
            this->participleSegment += " ";
        }
    }
    return true;
}

bool lifuren::Poetry::operator==(const lifuren::Poetry& poetry) const {
    if(this == &poetry) {
        return true;
    }
    return
        this->title      == poetry.title    &&
        this->author     == poetry.author   &&
        this->rhythmic   == poetry.rhythmic &&
        this->paragraphs == poetry.paragraphs;
}
