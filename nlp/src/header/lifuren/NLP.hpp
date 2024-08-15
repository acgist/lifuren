/**
 * 文档
 * 
 * @author acgist
 */
#ifndef LIFUREN_HEADER_NLP_NLP_HPP
#define LIFUREN_HEADER_NLP_NLP_HPP

#include "lifuren/RAG.hpp"
#include "lifuren/DocumentChunk.hpp"

namespace lifuren {

/**
 * RAG服务
 * 
 * 文档解析、文档分段、文档搜索
 */
class RAGService {

public:
    std::unique_ptr<lifuren::RAGClient> ragClient{ nullptr };
    std::unique_ptr<lifuren::ChunkService> chunkService{ nullptr };

};

} // END OF lifuren

#endif // END OF LIFUREN_HEADER_NLP_NLP_HPP
