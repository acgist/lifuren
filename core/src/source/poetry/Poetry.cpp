#include "../../header/Poetry.hpp"

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
    if(this->segment.empty() && !this->paragraphs.empty()) {
        for(
            auto iterator = this->paragraphs.begin();
            iterator != this->paragraphs.end();
            ++iterator
        ) {
            this->segment += lifuren::strings::trim(*iterator);
        }
    }
    return *this;
}

bool lifuren::Poetry::matchRule() {

    return true;
}

bool lifuren::Poetry::participleSegment() {
    return true;
}
