/**
 * https://github.com/facebookresearch/faiss/tree/main/tutorial/cpp
 */
#include "lifuren/RAG.hpp"

#include "spdlog/spdlog.h"

#include "faiss/IndexFlat.h"
#include "faiss/MetaIndexes.h"
// #include "faiss/IndexIDMap.h"

// 真实索引
static std::map<size_t, std::shared_ptr<faiss::Index>> indexBasicDBMap{};
// 映射索引
static std::map<size_t, std::shared_ptr<faiss::Index>> indexIdMapDBMap{};
// 使用内存地址作为内容ID
static std::map<size_t, std::shared_ptr<std::set<std::string*>>> idsMap{};

lifuren::FaissRAGClient::FaissRAGClient(const std::string& path, const std::string& embedding) : RAGClient(path, embedding) {
    if(idsMap.contains(this->id)) {
        this->ids = idsMap.at(this->id);
        this->indexBasicDB = indexBasicDBMap.at(this->id);
        this->indexIdMapDB = indexIdMapDBMap.at(this->id);
    } else {
        this->ids = std::make_shared<std::set<std::string*>>();
        this->indexBasicDB = std::make_shared<faiss::IndexFlatL2>(this->embeddingClient->getDims());
        this->indexIdMapDB = std::make_shared<faiss::IndexIDMap>(this->indexBasicDB.get());
        idsMap.emplace(this->id, this->ids);
        indexBasicDBMap.emplace(this->id, this->indexBasicDB);
        indexIdMapDBMap.emplace(this->id, this->indexIdMapDB);
    }
}

lifuren::FaissRAGClient::~FaissRAGClient() {
    if(!idsMap.contains(this->id)) {
        // 如果已经删除索引删除无效数据
        std::for_each(this->ids->begin(), this->ids->end(), [](const auto& ptr) {
            delete ptr;
        });
        this->ids->clear();
    }
}

std::vector<float> lifuren::FaissRAGClient::index(const std::string& content) {
    std::vector<float>&& vector = this->embeddingClient->getSegmentVector(content);
    std::string* ptr = new std::string(content);
    this->ids->insert(ptr);
    int64_t id = reinterpret_cast<size_t>(ptr);
    indexIdMapDB->add_with_ids(1, vector.data(), &id);
    return vector;
}

std::vector<std::string> lifuren::FaissRAGClient::search(const std::string& prompt, const int size) {
    const int faiss_n = 1;
    std::vector<float>&& vector = this->embeddingClient->getSegmentVector(prompt);
    // 数据
    std::vector<float> data;
    data.resize(faiss_n * size);
    // 索引
    std::vector<int64_t> idx;
    idx.resize(faiss_n * size);
    indexIdMapDB->search(faiss_n, vector.data(), size, data.data(), idx.data());
    // 结果
    std::vector<std::string> ret;
    ret.reserve(size);
    for(int index = 0; index < size; ++index) {
        std::string* ptr = reinterpret_cast<std::string*>(idx[index]);
        if(this->ids->contains(ptr)) {
            ret.push_back(*ptr);
        }
    }
    return ret;
}

bool lifuren::FaissRAGClient::deleteRAG() {
    SPDLOG_INFO("删除Faiss索引：{}", this->id);
    this->truncateIndex();
    std::for_each(this->ids->begin(), this->ids->end(), [](const auto& ptr) {
        delete ptr;
    });
    this->ids->clear();
    this->indexBasicDB->reset();
    this->indexIdMapDB->reset();
    idsMap.erase(this->id);
    indexBasicDBMap.erase(this->id);
    indexIdMapDBMap.erase(this->id);
    return true;
}
