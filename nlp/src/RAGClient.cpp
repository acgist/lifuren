#include "lifuren/RAGClient.hpp"

#include <set>
#include <mutex>
#include <atomic>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/EmbeddingClient.hpp"

// 锁
static std::mutex mutex;
// 索引计数
static std::atomic<int> share_count(0);
// 避免单次重复索引
static std::set<std::string> prompt_cache;

lifuren::RAGClient::RAGClient(
    const std::string& path,
    const std::string& embedding
) : path(path),
    embeddingClient(lifuren::EmbeddingClient::getClient(path, embedding))
{
    ++share_count;
}

lifuren::RAGClient::~RAGClient() {
    if(--share_count <= 0) {
        SPDLOG_DEBUG("没有引用清空提示索引缓存");
        std::lock_guard<std::mutex> lock(mutex);
        prompt_cache.clear();
    }
}

size_t lifuren::RAGClient::getDims() const {
    if(this->embeddingClient) {
        return this->embeddingClient->getDims();
    } else {
        return 0;
    }
}

bool lifuren::RAGClient::donePromptEmplace(const std::string& prompt) {
    std::lock_guard<std::mutex> lock(mutex);
    auto iterator = prompt_cache.find(prompt);
    if(iterator == prompt_cache.end()) {
        prompt_cache.insert(prompt);
        return false;
    }
    return true;
}

std::vector<std::string> lifuren::RAGClient::search(const std::string& prompt, const uint8_t size) const {
    return this->search(this->embeddingClient->getVector(prompt), size);
}

std::unique_ptr<lifuren::RAGClient> lifuren::RAGClient::getClient(const std::string& rag, const std::string& path, const std::string& embedding) {
    if(rag == "faiss" || rag == "Faiss") {
        return std::make_unique<lifuren::FaissRAGClient>(path, embedding);
    } else if(rag == "elasticsearch" || rag == "ElasticSearch") {
        return std::make_unique<lifuren::ElasticSearchRAGClient>(path, embedding);
    } else {
        SPDLOG_WARN("不支持的类型：{}", rag);
    }
    return nullptr;
}
