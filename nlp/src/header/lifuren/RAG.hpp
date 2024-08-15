/**
 * RAG
 */
#ifndef LIFUREN_HEADER_NLP_RAG_HPP
#define LIFUREN_HEADER_NLP_RAG_HPP

#include <string>
#include <vector>

#include "lifuren/Client.hpp"

namespace lifuren {

/**
 * 词向量服务
 */
class EmbeddingService {

public:
    // 词嵌入终端
    std::unique_ptr<EmbeddingClient> embeddingClient{ nullptr };

};

/**
 * RAG终端
 */
class RAGClient {

public:
    std::unique_ptr<lifuren::EmbeddingService> embeddingService{ nullptr };

public:
    /**
     * 索引建立
     */
    virtual std::vector<double> index(const std::string& content) = 0;
    /**
     * 索引搜索
     */
    virtual std::string search(const std::vector<double>& vector) = 0;

};

class Neo4jRAGClient : public RAGClient {
};

class FaissRAGClient : public RAGClient {
};

class MilvusRAGClient : public RAGClient {
};

class ChromaRAGClient : public RAGClient {
};

class TypesenseRAGClient : public RAGClient {
};

class ElasticSearchRAGClient : public RAGClient {
};

} // END OF lifuren

#endif // END OF LIFUREN_HEADER_NLP_RAG_HPP
