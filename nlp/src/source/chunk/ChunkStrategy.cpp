#include "lifuren/DocumentChunk.hpp"

lifuren::ChunkStrategy::ChunkStrategy(lifuren::ChunkType type) : type(type) {
}

lifuren::ChunkStrategy::~ChunkStrategy() {
}

lifuren::ChunkType lifuren::ChunkStrategy::chunkType() {
    return this->type;
}
