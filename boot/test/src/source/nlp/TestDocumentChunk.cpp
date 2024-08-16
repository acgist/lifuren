#include "lifuren/DocumentChunk.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

[[maybe_unused]] static void testLineChunk() {
    lifuren::PDFReader reader{ "D:/tmp/word.pdf" };
    // lifuren::TextReader reader{ "D:/tmp/lifuren.txt" };
    lifuren::LineChunkStrategy chunk{};
    while(reader.hasMore()) {
        const auto list = chunk.chunk(reader.readMore());
        for(const auto& line : list) {
            SPDLOG_DEBUG("line = {}", line);
        }
        SPDLOG_DEBUG("====more====");
    }
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testLineChunk();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
