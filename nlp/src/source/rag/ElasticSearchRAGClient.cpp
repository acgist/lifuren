#include "lifuren/RAG.hpp"

lifuren::ElasticSearchRAGClient::ElasticSearchRAGClient(const std::string& embedding) :RAGClient(embedding) {
}

lifuren::ElasticSearchRAGClient::~ElasticSearchRAGClient() {
}

std::vector<double> lifuren::ElasticSearchRAGClient::index(const std::string& content) {
    return {};
}

std::string lifuren::ElasticSearchRAGClient::search(const std::vector<double>& vector) {
    return "";
}
