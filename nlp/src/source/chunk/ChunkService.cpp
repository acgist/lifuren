#include "lifuren/DocumentChunk.hpp"

lifuren::ChunkService::ChunkService(const std::string& chunkType) {
    this->chunkStrategy = lifuren::ChunkStrategy::getChunkStrategy(chunkType);
}

lifuren::ChunkService::~ChunkService() {
}

std::list<std::string> lifuren::ChunkService::chunk(const std::string& path) {
    if(!this->chunkStrategy) {
        return {};
    }
    std::unique_ptr<lifuren::DocumentReader> documentReader{ lifuren::DocumentReader::getReader(path) };
    if(!documentReader) {
        return {};
    }
    std::list<std::string> ret;
    while(documentReader->hasMore()) {
        const auto&& more = documentReader->readMore();
        const bool   last = !documentReader->hasMore();
        auto&& list = this->chunkStrategy->chunk(more, last);
        ret.splice(ret.end(), list);
    }
    return ret;
}
