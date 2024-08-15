#include "lifuren/DocumentChunk.hpp"

lifuren::ChunkService::ChunkService(const std::string& chunkType) {
    if(chunkType == "LINE") {
        this->chunkStrategy = std::make_unique<lifuren::LineChunkStrategy>();
    } else {
        // TODO
    }
}

lifuren::ChunkService::~ChunkService() {
}

std::list<std::string> lifuren::ChunkService::chunk(const std::string& path) {
    // 文档读取
    std::unique_ptr<lifuren::DocumentReader> documentReader{ nullptr };
}
