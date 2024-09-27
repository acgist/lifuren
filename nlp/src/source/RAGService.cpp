#include "lifuren/RAG.hpp"

#include <mutex>

#include "spdlog/spdlog.h"

static std::recursive_mutex mutex;

lifuren::RAGService::RAGService() {
}

lifuren::RAGService::~RAGService() {
}

lifuren::RAGService& lifuren::RAGService::getInstance() {
    static lifuren::RAGService instance;
    return instance;
}

std::shared_ptr<lifuren::RAGTaskRunner> lifuren::RAGService::getRAGTask(const std::string& path) const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const auto iterator = this->taskMap.find(path);
    if(iterator == this->taskMap.end()) {
        return nullptr;
    }
    return iterator->second;
}

std::shared_ptr<lifuren::RAGTaskRunner> lifuren::RAGService::runRAGTask(const std::string& path) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const auto& rag       = lifuren::config::CONFIG.rag;
    const auto& embedding = lifuren::config::CONFIG.embedding;
    RAGTask task {
        .rag       = rag.type,
        .embedding = embedding.type,
        .path      = path,
    };
    const auto iterator = this->taskMap.find(path);
    if(iterator != this->taskMap.end()) {
        SPDLOG_DEBUG("RAG任务已经添加：{}", path);
        return iterator->second;
    }
    const auto runner = std::make_shared<lifuren::RAGTaskRunner>(task);
    if(runner->id <= 0LL || runner->stop || runner->finish) {
        SPDLOG_WARN("添加RAG任务失败：{}", path);
        return nullptr;
    } else if(runner->startExecute()) {
        SPDLOG_DEBUG("添加RAG任务：{}", path);
        const auto pair = this->taskMap.emplace(path, runner);
        return pair.first->second;
    } else {
        SPDLOG_WARN("执行RAG任务失败：{}", path);
        return nullptr;
    }
}

bool lifuren::RAGService::stopRAGTask(const std::string& path) const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const auto iterator = this->taskMap.find(path);
    if(iterator == this->taskMap.end()) {
        SPDLOG_DEBUG("RAG任务已经结束：{}", path);
        return true;
    }
    SPDLOG_DEBUG("结束RAG任务：{}", path);
    iterator->second->stop = true;
    return true;
}

bool lifuren::RAGService::removeRAGTask(const std::string& path) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const auto iterator = this->taskMap.find(path);
    if(iterator == this->taskMap.end()) {
        SPDLOG_DEBUG("RAG任务已经移除：{}", path);
        return true;
    }
    if(!iterator->second->stop && !iterator->second->finish) {
        SPDLOG_DEBUG("移除RAG任务失败：{} - {} - {}", path, iterator->second->stop, iterator->second->finish);
        return false;
    }
    SPDLOG_DEBUG("移除RAG任务：{}", path);
    return this->taskMap.erase(path) > 0;
}

size_t lifuren::RAGService::taskCount() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return this->taskMap.size();
}
