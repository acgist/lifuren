#include "lifuren/RAGClient.hpp"

#include <fstream>

#include "spdlog/spdlog.h"

#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/MetaIndexes.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/EmbeddingClient.hpp"

// 公用一个锁不考虑并发
static std::mutex mutex;
// 向量数量
const static int faiss_n = 1;

// 加载提示映射
static std::shared_ptr<std::map<size_t, std::string>> loadMapping(const std::filesystem::path& path);
// 保存提示映射
static void saveMapping(std::shared_ptr<std::map<size_t, std::string>> map, const std::filesystem::path& path);

// 加载Faiss向量库
static faiss::Index* loadIndexDB(size_t dims, const std::filesystem::path& path);
// 保存Faiss向量库
static void saveIndexDB(faiss::Index* index, const std::filesystem::path& path);

lifuren::FaissRAGClient::FaissRAGClient(
    const std::string& path,
    const std::string& embedding
) : RAGClient(path, embedding) {
    std::lock_guard<std::mutex> lock(mutex);
    const std::filesystem::path indexDBPath = lifuren::file::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::INDEXDB_MODEL_FILE });
    this->indexDB.reset(loadIndexDB(this->embeddingClient->getDims(), indexDBPath));
    const std::filesystem::path mappingPath = lifuren::file::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::MAPPING_MODEL_FILE });
    this->mapping = loadMapping(mappingPath);
}

lifuren::FaissRAGClient::~FaissRAGClient() {
    std::lock_guard<std::mutex> lock(mutex);
    const std::filesystem::path indexDBPath = lifuren::file::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::INDEXDB_MODEL_FILE });
    saveIndexDB(this->indexDB.get(), indexDBPath);
    const std::filesystem::path mappingPath = lifuren::file::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::MAPPING_MODEL_FILE });
    saveMapping(this->mapping, mappingPath);
}

std::vector<float> lifuren::FaissRAGClient::index(const std::string& prompt) {
    const std::vector<float> vector = std::move(this->embeddingClient->getVector(prompt));
    if(this->donePromptEmplace(prompt)) {
        return vector;
    }
    if(vector.empty()) {
        return vector;
    }
    const int64_t id = lifuren::config::uuid();
    if(!this->mapping || !this->indexDB) {
        SPDLOG_WARN("没有初始化");
        return vector;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        this->mapping->emplace(id, prompt);
        this->indexDB->add_with_ids(faiss_n, vector.data(), &id);
    }
    return vector;
}

std::vector<std::string> lifuren::FaissRAGClient::search(const std::vector<float>& prompt, const uint8_t size) const {
    std::vector<float> distance;
    distance.resize(faiss_n * size);
    std::vector<int64_t> idx;
    idx.resize(faiss_n * size);
    {
        std::lock_guard<std::mutex> lock(mutex);
        this->indexDB->search(faiss_n, prompt.data(), size, distance.data(), idx.data());
    }
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

static std::shared_ptr<std::map<size_t, std::string>> loadMapping(const std::filesystem::path& path) {
    std::shared_ptr<std::map<size_t, std::string>> map = std::make_shared<std::map<size_t, std::string>>();
    if(!std::filesystem::exists(path)) {
        return map;
    }
    std::ifstream stream;
    stream.open(path, std::ios_base::in | std::ios_base::binary);
    if(stream.is_open()) {
        size_t  id     = 0;
        uint8_t length = 0;
        while(
            stream.read(reinterpret_cast<char*>(&id),     sizeof(id)) &&
            stream.read(reinterpret_cast<char*>(&length), sizeof(length))
        ) {
            std::string word;
            word.resize(length);
            if(stream.read(word.data(), length)) {
                map->emplace(id, std::move(word));
            } else {
                break;
            }
        }
        SPDLOG_DEBUG("加载映射文件：{}", map->size());
    } else {
        SPDLOG_WARN("Faiss映射文件打开失败：{}", path.string());
    }
    stream.close();
    return map;
}

static void saveMapping(std::shared_ptr<std::map<size_t, std::string>> map, const std::filesystem::path& path) {
    SPDLOG_DEBUG("保存映射文件：{}", path.string());
    lifuren::file::createFolder(path.parent_path());
    std::ofstream stream;
    stream.open(path, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if(stream.is_open()) {
        for(const auto& [id, word] : *map) {
            const uint8_t size = static_cast<uint8_t>(word.size());
            stream.write(reinterpret_cast<const char*>(&id), sizeof(id));
            stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
            stream.write(word.data(), word.size());
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
    SPDLOG_DEBUG("保存索引文件：{}", path.string());
    lifuren::file::createFolder(path.parent_path());
    faiss::write_index(index, path.string().c_str());
}
