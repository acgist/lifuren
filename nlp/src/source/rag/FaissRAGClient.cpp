/**
 * https://github.com/facebookresearch/faiss/tree/main/tutorial/cpp
 */
#include "lifuren/RAG.hpp"

#include "spdlog/spdlog.h"

#include "faiss/IndexFlat.h"
#include "faiss/MetaIndexes.h"
// #include "faiss/IndexIDMap.h"

// 查询索引数量
static const int faiss_n = 1;
// 使用内存地址作为内容ID
static std::set<std::string*> ids{};

lifuren::FaissRAGClient::FaissRAGClient(const std::string& path, const std::string& embedding) : RAGClient(path, embedding) {
    this->indexBasicDB = std::make_unique<faiss::IndexFlatL2>(this->embeddingClient->getDims());
    this->indexIdMapDB = std::make_unique<faiss::IndexIDMap>(this->indexBasicDB.get());
}

lifuren::FaissRAGClient::~FaissRAGClient() {
}

std::vector<float> lifuren::FaissRAGClient::index(const std::string& content) {
    std::vector<float>&& vector = this->embeddingClient->getSegmentVector(content);
    std::string* ptr = new std::string(content);
    ids.insert(ptr);
    faiss::idx_t id = reinterpret_cast<size_t>(ptr);
    this->indexIdMapDB->add_with_ids(faiss_n, vector.data(), &id);
    return vector;
}

std::vector<std::string> lifuren::FaissRAGClient::search(const std::string& prompt, const int size) {
    std::vector<float>&& vector = this->embeddingClient->getSegmentVector(prompt);
    // 数据
    std::vector<float> data;
    data.resize(faiss_n * size);
    // 索引
    std::vector<faiss::idx_t> idx;
    idx.resize(faiss_n * size);
    this->indexIdMapDB->search(faiss_n, vector.data(), size, data.data(), idx.data());
    // 结果
    std::vector<std::string> ret;
    ret.reserve(size);
    for(int index = 0; index < size; ++index) {
        std::string* ptr = reinterpret_cast<std::string*>(idx[index]);
        if(ids.contains(ptr)) {
            ret.push_back(*ptr);
        }
    }
    return ret;
}

bool lifuren::FaissRAGClient::deleteRAG() {
    SPDLOG_INFO("删除Faiss索引：{}", this->id);
    this->truncateIndex();
    for(const auto& ptr : ids) {
        delete ptr;
    }
    ids.clear();
    return true;
}
