#include "lifuren/DocumentChunk.hpp"

#include "spdlog/spdlog.h"

lifuren::ChunkStrategy::ChunkStrategy(lifuren::ChunkType type) : type(type) {
}

lifuren::ChunkStrategy::~ChunkStrategy() {
}

lifuren::ChunkType lifuren::ChunkStrategy::chunkType() {
    return this->type;
}

std::unique_ptr<lifuren::ChunkStrategy> lifuren::ChunkStrategy::getChunkStrategy(const std::string& chunkType) {
    if(chunkType == "LINE") {
        return std::make_unique<lifuren::LineChunkStrategy>();
    } else {
        SPDLOG_WARN("不支持的分段模式：{}", chunkType);
    }
    return nullptr;
}
