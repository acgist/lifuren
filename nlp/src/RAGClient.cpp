#include "lifuren/RAGClient.hpp"

#include <set>
#include <mutex>
#include <atomic>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/EmbeddingClient.hpp"

// 避免单次重复索引
static std::set<std::string> promptCache;
// 索引计数
static std::atomic<int> share_count(0);
// 锁
static std::mutex mutex;

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
        std::lock_guard<std::mutex> lock(mutex);
        promptCache.clear();
        SPDLOG_DEBUG("没有引用清空提示索引缓存");
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
    auto iterator = promptCache.find(prompt);
    if(iterator == promptCache.end()) {
        promptCache.insert(prompt);
        return false;
    }
    return true;
}

std::vector<std::string> lifuren::RAGClient::search(const std::string& prompt, const uint8_t size) const {
    return this->search(std::move(this->embeddingClient->getVector(prompt)), size);
}

std::unique_ptr<lifuren::RAGClient> lifuren::RAGClient::getClient(const std::string& rag, const std::string& path, const std::string& embedding) {
    if(rag == "faiss" || rag == "Faiss") {
        return std::make_unique<lifuren::FaissRAGClient>(path, embedding);
    } else if(rag == "elasticsearch" || rag == "ElasticSearch") {
        return std::make_unique<lifuren::ElasticSearchRAGClient>(path, embedding);
    } else {
        SPDLOG_WARN("不支持的RAGClient类型：{}", rag);
    }
    return nullptr;
}
