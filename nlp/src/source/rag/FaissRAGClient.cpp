/**
 * https://github.com/facebookresearch/faiss/tree/main/tutorial/cpp
 */
#include "lifuren/RAGClient.hpp"

#include "faiss/IndexFlat.h"
#include "faiss/MetaIndexes.h"
// #include "faiss/IndexIDMap.h"

lifuren::FaissRAGClient::FaissRAGClient(const std::string& path, const std::string& embedding) : RAGClient(path, embedding) {
    this->indexBasicDB = std::make_unique<faiss::IndexFlatL2>(this->embeddingClient->getDims());
    this->indexIdMapDB = std::make_unique<faiss::IndexIDMap>(this->indexBasicDB.get());
}

lifuren::FaissRAGClient::~FaissRAGClient() {
}

std::vector<float> lifuren::FaissRAGClient::index(const std::string& content) {
    std::vector<float>&& vector = this->embeddingClient->getSegmentVector(content, lifuren::EmbeddingClient::SegmentType::CHAR);
    // TODO: 加入index
    return vector;
}

std::vector<std::string> lifuren::FaissRAGClient::search(const std::string& prompt) {
    std::vector<float>&& vector = this->embeddingClient->getSegmentVector(prompt, lifuren::EmbeddingClient::SegmentType::CHAR);
    // TODO: 搜索index
    // this->indexIdMapDB->search()
    return {};
}

bool lifuren::FaissRAGClient::deleteRAG() {
    return true;
}
