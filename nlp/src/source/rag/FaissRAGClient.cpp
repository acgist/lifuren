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

static std::map<size_t, std::shared_ptr<faiss::Index>> indexDBMap{};
// TODO: 使用short代替size_t
static std::map<size_t, std::shared_ptr<std::map<size_t, std::string>>> mappingMap{};

static std::shared_ptr<std::map<size_t, std::string>> loadMapping(const std::filesystem::path& path);
static void saveMapping(std::shared_ptr<std::map<size_t, std::string>> map, const std::filesystem::path& path);

static faiss::Index* loadIndexDB(size_t dims, const std::filesystem::path& path);
static void saveIndexDB(faiss::Index* index, const std::filesystem::path& path);

lifuren::FaissRAGClient::FaissRAGClient(const std::string& path, const std::string& embedding) : RAGClient(path, embedding) {
}

lifuren::FaissRAGClient::~FaissRAGClient() {
}

std::vector<float> lifuren::FaissRAGClient::index(const std::string& prompt) {
    // TODO: 是否需要重复验证
    // TODO: 验证一秒钟会不会超过一万
    const int faiss_n = 1;
    const int64_t id = lifuren::uuid();
    const std::vector<float>&& vector = this->embeddingClient->getVector(prompt);
    this->mapping->emplace(id, prompt);
    this->indexDB->add_with_ids(faiss_n, vector.data(), &id);
    return vector;
}

std::vector<std::string> lifuren::FaissRAGClient::search(const std::vector<float>& prompt, const int size) const {
    const int faiss_n = 1;
    std::vector<float> data;
    data.resize(faiss_n * size);
    std::vector<int64_t> idx;
    idx.resize(faiss_n * size);
    this->indexDB->search(faiss_n, prompt.data(), size, data.data(), idx.data());
    std::vector<std::string> ret;
    ret.reserve(size);
    for(int index = 0; index < size; ++index) {
        auto iterator = this->mapping->find(idx[index]);
        if(iterator == this->mapping->end()) {
            SPDLOG_WARN("索引没有映射：{}", idx[index]);
        } else {
            ret.push_back(iterator->second);
        }
    }
    return ret;
}

void lifuren::FaissRAGClient::loadIndex() {
    lifuren::RAGClient::loadIndex();
    // TODO: 异步加载
    if(mappingMap.contains(this->id)) {
        this->mapping = mappingMap.at(this->id);
        this->indexDB = indexDBMap.at(this->id);
    } else {
        const std::filesystem::path indexDBPath = lifuren::files::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::INDEXDB_MODEL_FILE });
        this->indexDB.reset(loadIndexDB(this->embeddingClient->getDims(), indexDBPath));
        indexDBMap.emplace(this->id, this->indexDB);
        const std::filesystem::path mappingPath = lifuren::files::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::MAPPING_MODEL_FILE });
        this->mapping = loadMapping(mappingPath);
        mappingMap.emplace(this->id, this->mapping);
    }
}

void lifuren::FaissRAGClient::saveIndex() {
    lifuren::RAGClient::saveIndex();
    const std::filesystem::path indexDBPath = lifuren::files::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::INDEXDB_MODEL_FILE });
    saveIndexDB(this->indexDB.get(), indexDBPath);
    const std::filesystem::path mappingPath = lifuren::files::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::MAPPING_MODEL_FILE });
    saveMapping(this->mapping, mappingPath);
}

void lifuren::FaissRAGClient::truncateIndex() {
    this->mapping->clear();
    this->indexDB->reset();
    mappingMap.erase(this->id);
    indexDBMap.erase(this->id);
    lifuren::RAGClient::truncateIndex();
}

static std::shared_ptr<std::map<size_t, std::string>> loadMapping(const std::filesystem::path& path) {
    std::shared_ptr<std::map<size_t, std::string>> map = std::make_shared<std::map<size_t, std::string>>();
    if(!std::filesystem::exists(path)) {
        return map;
    }
    std::ifstream stream;
    stream.open(path, std::ios_base::in | std::ios_base::binary);
    if(stream.is_open()) {
        size_t  id     = 0LL;
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
    return map;
}

static void saveMapping(std::shared_ptr<std::map<size_t, std::string>> map, const std::filesystem::path& path) {
    lifuren::files::createFolder(path.parent_path());
    std::ofstream stream;
    stream.open(path, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if(stream.is_open()) {
        for(const auto& [id, word] : *map) {
            stream << id;
            stream << static_cast<uint8_t>(word.size());
            stream << word;
        }
    } else {
        SPDLOG_WARN("Faiss映射文件打开失败：{}", path.string());
    }
    stream.close();
}

static faiss::Index* loadIndexDB(size_t dims, const std::filesystem::path& path) {
    if(std::filesystem::exists(path)) {
        auto index = faiss::read_index(path.string().c_str());
        if(index) {
            return index;
        } else {
            SPDLOG_WARN("Faiss索引文件打开失败：{}", path.string());
        }
    }
    return new faiss::IndexIDMap(new faiss::IndexFlatL2(dims));
}

static void saveIndexDB(faiss::Index* index, const std::filesystem::path& path) {
    lifuren::files::createFolder(path.parent_path());
    faiss::write_index(index, path.string().c_str());
}
