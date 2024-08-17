#include "lifuren/DocumentChunk.hpp"

lifuren::LineChunkStrategy::LineChunkStrategy() : ChunkStrategy(lifuren::ChunkType::LINE) {
}

lifuren::LineChunkStrategy::~LineChunkStrategy() {
}

std::list<std::string> lifuren::LineChunkStrategy::chunk(const std::string& content, bool last) {
    this->document += content;
    std::list<std::string> list;
    size_t pos = this->document.find('\n');
    while(pos != std::string::npos) {
        list.emplace_back(this->document.substr(0, pos));
        this->document = this->document.substr(pos + 1);
        pos = this->document.find('\n');
    }
    if(last && !this->document.empty()) {
        list.emplace_back(this->document);
    }
    return list;
}
