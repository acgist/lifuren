/**
 * https://github.com/facebookresearch/faiss/tree/main/tutorial/cpp
 */
#include "lifuren/RAG.hpp"

#include "spdlog/spdlog.h"

#include "faiss/IndexFlat.h"
#include "faiss/MetaIndexes.h"

#include "lifuren/Lifuren.hpp"

// 真实索引
static std::map<size_t, std::shared_ptr<faiss::Index>> indexBasicDBMap{};
// 映射索引
static std::map<size_t, std::shared_ptr<faiss::Index>> indexIdMapDBMap{};
// 使用内存地址作为内容ID
static std::map<size_t, std::shared_ptr<std::map<size_t, std::string>>> idMappingMap{};

lifuren::FaissRAGClient::FaissRAGClient(const std::string& path, const std::string& embedding) : RAGClient(path, embedding) {
    if(idMappingMap.contains(this->id)) {
        this->idMapping = idMappingMap.at(this->id);
        this->indexBasicDB = indexBasicDBMap.at(this->id);
        this->indexIdMapDB = indexIdMapDBMap.at(this->id);
    } else {
        this->idMapping = std::make_shared<std::map<size_t, std::string>>();
        this->indexBasicDB = std::make_shared<faiss::IndexFlatL2>(this->embeddingClient->getDims());
        this->indexIdMapDB = std::make_shared<faiss::IndexIDMap>(this->indexBasicDB.get());
        idMappingMap.emplace(this->id, this->idMapping);
        indexBasicDBMap.emplace(this->id, this->indexBasicDB);
        indexIdMapDBMap.emplace(this->id, this->indexIdMapDB);
    }
}

lifuren::FaissRAGClient::~FaissRAGClient() {
}

std::vector<float> lifuren::FaissRAGClient::index(const std::string& content) {
    // TODO: 验证一秒钟会不会超过一万
    const int faiss_n = 1;
    const int64_t id = lifuren::uuid();
    std::vector<float>&& vector = this->embeddingClient->getSegmentVector(content);
    this->idMapping->emplace(id, content);
    this->indexIdMapDB->add_with_ids(faiss_n, vector.data(), &id);
    return vector;
}

std::vector<std::string> lifuren::FaissRAGClient::search(const std::vector<float>& prompt, const int size) {
    const int faiss_n = 1;
    // 数据
    std::vector<float> data;
    data.resize(faiss_n * size);
    // 索引
    std::vector<int64_t> idx;
    idx.resize(faiss_n * size);
    this->indexIdMapDB->search(faiss_n, prompt.data(), size, data.data(), idx.data());
    // 结果
    std::vector<std::string> ret;
    ret.reserve(size);
    for(int index = 0; index < size; ++index) {
        auto iterator = this->idMapping->find(idx[index]);
        if(iterator == this->idMapping->end()) {
            SPDLOG_WARN("索引没有映射：{}", idx[index]);
        } else {
            ret.push_back(iterator->second);
        }
    }
    return ret;
}

void lifuren::FaissRAGClient::truncateIndex() {
    lifuren::RAGClient::truncateIndex();
    this->idMapping->clear();
    this->indexBasicDB->reset();
    this->indexIdMapDB->reset();
    idMappingMap.erase(this->id);
    indexBasicDBMap.erase(this->id);
    indexIdMapDBMap.erase(this->id);
}
