#include "lifuren/Poetry.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/String.hpp"
#include "lifuren/EmbeddingClient.hpp"

lifuren::poetry::Poetry& lifuren::poetry::Poetry::preproccess() {
    if(this->title.empty() && !this->rhythm.empty()) {
        this->title = this->rhythm;
    } else if(!this->title.empty() && this->rhythm.empty()) {
        this->rhythm = this->title;
    } else {
        //
    }
    const std::string&& content = lifuren::string::join(this->paragraphs, "");
    this->simpleParagraphs = lifuren::string::split(content, lifuren::poetry::POETRY_SIMPLE);
    this->segment = lifuren::string::join(this->paragraphs, "\n");
    this->simpleSegment = lifuren::string::join(this->simpleParagraphs, "\n");
    return *this;
}

bool lifuren::poetry::Poetry::matchRhythm() {
    if(this->rhythmPtr) {
        return true;
    }
    std::vector<uint32_t> segmentRule(this->simpleParagraphs.size());
    std::transform(this->simpleParagraphs.begin(), this->simpleParagraphs.end(), segmentRule.begin(), [](const auto& v) -> uint32_t {
        return static_cast<uint32_t>(lifuren::string::length(v));
    });
    auto iterator  = lifuren::config::RHYTHM.begin();
    const auto end = lifuren::config::RHYTHM.end();
    for(; iterator != end; ++iterator) {
        lifuren::config::Rhythm& rhythmRef = iterator->second;
        if(
            rhythmRef.segmentSize == static_cast<int>(this->simpleParagraphs.size()) &&
            rhythmRef.segmentRule == segmentRule
        ) {
            this->rhythmPtr = &rhythmRef;
            if(this->title.empty()) {
                this->title = rhythmRef.title;
            }
            if(this->rhythm.empty()) {
                this->rhythm = rhythmRef.rhythm;
            }
            break;
        }
    }
    return this->rhythmPtr;
}

bool lifuren::poetry::Poetry::participle() {
    if(this->rhythmPtr == nullptr) {
        return false;
    }
    std::string word;
    uint32_t pos = 0;
    uint32_t len = 0;
    const std::vector<uint32_t>& participleRuleRef = this->rhythmPtr->participleRule;
    auto iterator  = participleRuleRef.begin();
    const auto end = participleRuleRef.end();
    auto paragraphsIterator  = this->simpleParagraphs.begin();
    const auto paragraphsEnd = this->simpleParagraphs.end();
    for(; iterator != end; ++iterator) {
        word = lifuren::string::substr(paragraphsIterator->c_str(), pos, *iterator);
        pos += *iterator;
        len += word.length();
        this->participleParagraphs.push_back(word);
        if(this->participleSegment.empty()) {
            this->participleSegment = word;
        } else {
            this->participleSegment = this->participleSegment + word;
        }
        if(len >= paragraphsIterator->length()) {
            pos = 0;
            len = 0;
            ++paragraphsIterator;
            if(paragraphsIterator == paragraphsEnd) {
                break;
            }
            this->participleSegment += "\n";
        } else {
            this->participleSegment += " ";
        }
    }
    return true;
}

bool lifuren::poetry::Poetry::operator==(const lifuren::poetry::Poetry& poetry) const {
    if(this == &poetry) {
        return true;
    }
    // 内容相同即可
    return this->paragraphs == poetry.paragraphs;
}
