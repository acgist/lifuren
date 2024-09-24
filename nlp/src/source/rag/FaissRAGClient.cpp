/**
 * https://github.com/facebookresearch/faiss/tree/main/tutorial/cpp
 */
#include "lifuren/RAG.hpp"

#include <fstream>

#include "spdlog/spdlog.h"

#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/MetaIndexes.h"

#include "lifuren/Files.hpp"
#include "lifuren/Lifuren.hpp"

static std::map<size_t, std::shared_ptr<faiss::Index>> indexIdMapDBMap{};
static std::map<size_t, std::shared_ptr<std::map<size_t, std::string>>> idMappingMap{};

static void serialization(std::shared_ptr<std::map<size_t, std::string>> map, const std::filesystem::path& path);
static void unserialization(std::shared_ptr<std::map<size_t, std::string>> map, const std::filesystem::path& path);

lifuren::FaissRAGClient::FaissRAGClient(const std::string& path, const std::string& embedding) : RAGClient(path, embedding) {
}

lifuren::FaissRAGClient::~FaissRAGClient() {
}

std::vector<float> lifuren::FaissRAGClient::index(const std::string& content) {
    // TODO: 是否需要重复验证
    // TODO: 验证一秒钟会不会超过一万
    const int faiss_n = 1;
    const int64_t id = lifuren::uuid();
    std::vector<float>&& vector = this->embeddingClient->getVector(content);
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

void lifuren::FaissRAGClient::loadIndex() {
    lifuren::RAGClient::loadIndex();
    if(idMappingMap.contains(this->id)) {
        this->idMapping = idMappingMap.at(this->id);
        this->indexIdMapDB = indexIdMapDBMap.at(this->id);
    } else {
        // indexIdMapDB
        const std::filesystem::path faissPath = lifuren::files::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::FAISS_MODEL_FILE });
        if(std::filesystem::exists(faissPath)) {
            auto index = faiss::read_index(faissPath.string().c_str());
            if(index) {
                this->indexIdMapDB.reset(index);
            } else {
                SPDLOG_WARN("加载本地Faiss索引失败：{}", faissPath.string());
            }
        }
        if(!this->indexIdMapDB) {
            this->indexIdMapDB = std::make_shared<faiss::IndexIDMap>(new faiss::IndexFlatL2(this->embeddingClient->getDims()));
        }
        indexIdMapDBMap.emplace(this->id, this->indexIdMapDB);
        // idMapping
        const std::filesystem::path mappingPath = lifuren::files::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::MAPPING_MODEL_FILE });
        this->idMapping = std::make_shared<std::map<size_t, std::string>>();
        unserialization(this->idMapping, mappingPath);
        idMappingMap.emplace(this->id, this->idMapping);
    }
}

void lifuren::FaissRAGClient::saveIndex() {
    lifuren::RAGClient::saveIndex();
    const std::filesystem::path faissPath   = lifuren::files::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::FAISS_MODEL_FILE });
    const std::filesystem::path mappingPath = lifuren::files::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::MAPPING_MODEL_FILE });
    faiss::write_index(this->indexIdMapDB.get(), faissPath.string().c_str());
    serialization(this->idMapping, mappingPath);
}

void lifuren::FaissRAGClient::truncateIndex() {
    this->idMapping->clear();
    this->indexIdMapDB->reset();
    idMappingMap.erase(this->id);
    indexIdMapDBMap.erase(this->id);
    lifuren::RAGClient::truncateIndex();
}

static void serialization(std::shared_ptr<std::map<size_t, std::string>> map, const std::filesystem::path& path) {
    lifuren::files::createFolder(path.parent_path());
    std::ofstream stream;
    stream.open(path, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if(stream.is_open()) {
        for(const auto& [id, word] : *map) {
            stream << id;
            uint8_t length = word.size();
            stream << length;
            stream << word;
        }
    } else {
        SPDLOG_WARN("Faiss映射文件打开失败：{}", path.string());
    }
    stream.close();
}

static void unserialization(std::shared_ptr<std::map<size_t, std::string>> map, const std::filesystem::path& path) {
    if(std::filesystem::exists(path)) {
        std::ifstream stream;
        stream.open(path, std::ios_base::in | std::ios_base::binary);
        if(stream.is_open()) {
            size_t id = 0LL;
            uint8_t length = 0;
            while(stream >> id && stream >> length) {
                std::string word;
                word.resize(length);
                if(stream.read(word.data(), length)) {
                    map->emplace(id, std::move(word));
                } else {
                    break;
                }
            }
        } else {
            SPDLOG_WARN("Faiss映射文件打开失败：{}", path.string());
        }
        stream.close();
    } else {
    }
}