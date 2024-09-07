/**
 * https://github.com/facebookresearch/faiss/tree/main/tutorial/cpp
 */
#include "lifuren/RAGClient.hpp"

#include "faiss/IndexFlat.h"

lifuren::FaissRAGClient::FaissRAGClient(const std::string& path, const std::string& embedding) : RAGClient(path, embedding) {
}

lifuren::FaissRAGClient::~FaissRAGClient() {
}

std::vector<float> lifuren::FaissRAGClient::index(const std::string& content) {
    return {};
}

std::vector<std::string> lifuren::FaissRAGClient::search(const std::string& prompt) {
    return {};
}

bool lifuren::FaissRAGClient::deleteRAG() {
    return true;
}
