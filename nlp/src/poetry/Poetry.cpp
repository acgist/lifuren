#include "lifuren/poetry/Poetry.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/String.hpp"
#include "lifuren/EmbeddingClient.hpp"

lifuren::poetry::Poetry& lifuren::poetry::Poetry::preproccess() {
    if(this->title.empty() && !this->rhythmic.empty()) {
        this->title = this->rhythmic;
    }
    if(!this->title.empty() && this->rhythmic.empty()) {
        this->rhythmic = this->title;
    }
    const std::string content = std::move(lifuren::string::join(this->paragraphs, ""));
    this->simpleParagraphs    = std::move(lifuren::string::split(content, lifuren::poetry::POETRY_SIMPLE));
    this->segment             = std::move(lifuren::string::join(this->paragraphs, "\n"));
    this->simpleSegment       = std::move(lifuren::string::join(this->simpleParagraphs, "\n"));
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
          auto beg = lifuren::config::RHYTHM.begin();
    const auto end = lifuren::config::RHYTHM.end();
    for(; beg != end; ++beg) {
        const lifuren::config::Rhythm& rhythmRef = beg->second;
        if(
            rhythmRef.segmentSize == static_cast<int>(this->simpleParagraphs.size()) &&
            rhythmRef.segmentRule == segmentRule
        ) {
            this->rhythmPtr = &rhythmRef;
            if(this->title.empty()) {
                this->title = rhythmRef.title;
            }
            if(this->rhythmic.empty()) {
                this->rhythmic = rhythmRef.rhythm;
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
    uint32_t pos = 0;
    uint32_t len = 0;
    std::string word;
    const std::vector<uint32_t>& participleRuleRef = this->rhythmPtr->participleRule;
          auto beg = participleRuleRef.begin();
    const auto end = participleRuleRef.end();
          auto paragraphsBeg = this->simpleParagraphs.begin();
    const auto paragraphsEnd = this->simpleParagraphs.end();
    for(; beg != end; ++beg) {
        word = std::move(lifuren::string::substr(paragraphsBeg->c_str(), pos, *beg));
        pos += *beg;
        len += word.length();
        this->participleParagraphs.push_back(word);
        if(this->participleSegment.empty()) {
            this->participleSegment = word;
        } else {
            this->participleSegment = this->participleSegment + word;
        }
        if(len >= paragraphsBeg->length()) {
            pos = 0;
            len = 0;
            ++paragraphsBeg;
            if(paragraphsBeg == paragraphsEnd) {
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
