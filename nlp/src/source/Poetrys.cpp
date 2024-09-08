#include "lifuren/Poetrys.hpp"

#include "lifuren/Strings.hpp"
#include "lifuren/model/Poetry.hpp"

std::vector<std::string> lifuren::poetrys::toChars(const std::string& poetry) {
    return lifuren::strings::toChars(replaceSymbol(poetry));
}

std::vector<std::string> lifuren::poetrys::toWords(const std::string& poetry) {
    lifuren::Poetry value;
    value.paragraphs = { poetry };
    value.preproccess();
    value.matchLabel();
    value.participle();
    return value.participleParagraphs;
}

std::vector<std::string> lifuren::poetrys::toSegments(const std::string& poetry) {
    return lifuren::strings::split(poetry, lifuren::poetry::POETRY_SEGMENT_DELIM);
}

std::string lifuren::poetrys::replaceSymbol(const std::string& poetry) {
    std::string copy = poetry;
    lifuren::strings::replace(copy, lifuren::poetry::POETRY_SYMBOL_DELIM, "");
    return copy;
}
