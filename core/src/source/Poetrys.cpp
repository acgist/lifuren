#include "lifuren/Poetrys.hpp"

#include "lifuren/Config.hpp"
#include "lifuren/Strings.hpp"

std::string lifuren::poetrys::beautify(const std::string& segment) {
    std::string ret;
    if(segment.empty()) {
        return ret;
    }
    std::vector<std::string> vector = lifuren::strings::split(segment, lifuren::poetrys::POETRY_BEAUTIFY_DELIM, true);
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

std::vector<std::string> lifuren::poetrys::toChars(const std::string& poetry) {
    return lifuren::strings::toChars(replaceSymbol(poetry));
}

std::vector<std::string> lifuren::poetrys::toWords(const std::string& poetry) {
    lifuren::poetrys::Poetry value;
    value.paragraphs = { poetry };
    value.preproccess();
    value.matchRhythm();
    value.participle();
    return value.participleParagraphs;
}

std::vector<std::string> lifuren::poetrys::toSegments(const std::string& poetry) {
    return lifuren::strings::split(poetry, lifuren::poetrys::POETRY_SEGMENT_DELIM);
}

std::string lifuren::poetrys::replaceSymbol(const std::string& poetry) {
    std::string copy = poetry;
    lifuren::strings::replace(copy, lifuren::poetrys::POETRY_SYMBOL_DELIM, "");
    return copy;
}

lifuren::poetrys::Poetry& lifuren::poetrys::Poetry::preproccess() {
    if(this->title.empty() && !this->rhythm.empty()) {
        this->title = this->rhythm;
    } else if(!this->title.empty() && this->rhythm.empty()) {
        this->rhythm = this->title;
    } else {
        //
    }
    std::string content = lifuren::strings::join(this->paragraphs, "");
    this->paragraphs = lifuren::strings::split(
        content,
        lifuren::poetrys::POETRY_BEAUTIFY_DELIM,
        true
    );
    this->simpleParagraphs = lifuren::strings::split(
        content,
        lifuren::poetrys::POETRY_SEGMENT_DELIM
    );
    this->segment = lifuren::strings::join(this->paragraphs, "\n");
    this->simpleSegment = lifuren::strings::join(this->simpleParagraphs, "\n");
    return *this;
}

bool lifuren::poetrys::Poetry::matchRhythm() {
    if(this->rhythmPtr) {
        return true;
    }
    std::vector<uint32_t> segmentRule(this->simpleParagraphs.size());
    std::transform(this->simpleParagraphs.begin(), this->simpleParagraphs.end(), segmentRule.begin(), [](auto& v) -> uint32_t {
        return (uint32_t) lifuren::strings::length(v);
    });
    for(
        auto iterator = lifuren::config::RHYTHM.begin();
        iterator != lifuren::config::RHYTHM.end();
        ++iterator
    ) {
        lifuren::config::Rhythm& ruythmRef = iterator->second;
        // TODO: 验证词能否正确匹配或者添加词牌
        if(
            ruythmRef.segmentSize == static_cast<int>(this->simpleParagraphs.size()) &&
            ruythmRef.segmentRule == segmentRule
        ) {
            this->rhythmPtr = &ruythmRef;
            // 不全词牌名称
            if(this->title.empty()) {
                this->title = ruythmRef.name;
            }
            if(this->rhythm.empty()) {
                this->rhythm = ruythmRef.rhythm;
            }
            break;
        }
    }
    return this->rhythmPtr;
}

bool lifuren::poetrys::Poetry::participle() {
    if(this->rhythmPtr == nullptr) {
        return false;
    }
    std::vector<uint32_t>& participleRuleRef = this->rhythmPtr->participleRule;
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

bool lifuren::poetrys::Poetry::operator==(const lifuren::poetrys::Poetry& poetry) const {
    if(this == &poetry) {
        return true;
    }
    return
        this->title      == poetry.title  &&
        this->author     == poetry.author &&
        this->rhythm     == poetry.rhythm &&
        this->paragraphs == poetry.paragraphs;
}
