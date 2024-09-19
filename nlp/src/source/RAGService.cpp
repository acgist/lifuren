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

std::shared_ptr<lifuren::RAGTaskRunner> lifuren::RAGService::getRAGTask(const std::string& path) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const auto iterator = this->tasks.find(path);
    if(iterator == this->tasks.end()) {
        return nullptr;
    }
    return iterator->second;
}

std::shared_ptr<lifuren::RAGTaskRunner> lifuren::RAGService::runRAGTask(const std::string& path) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const auto& rag       = lifuren::config::CONFIG.rag;
    const auto& embedding = lifuren::config::CONFIG.embedding;
    RAGTask task{
        .type      = rag.type,
        .path      = path,
        .embedding = embedding.type,
    };
    const auto iterator = this->tasks.find(task.path);
    if(iterator != this->tasks.end()) {
        SPDLOG_DEBUG("RAG任务已经添加：{}", task.path);
        return iterator->second;
    }
    const auto runner = std::make_shared<lifuren::RAGTaskRunner>(task);
    if(runner->id <= 0L) {
        SPDLOG_WARN("RAG任务没有索引标识：{}", task.path);
        return nullptr;
    } else {
        SPDLOG_DEBUG("添加RAG任务：{}", task.path);
        runner->startExecute();
        const auto pair = this->tasks.emplace(task.path, runner);
        return pair.first->second;
    }
}

bool lifuren::RAGService::stopRAGTask(const std::string& path) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto iterator = this->tasks.find(path);
    if(iterator == this->tasks.end()) {
        SPDLOG_DEBUG("RAG任务已经结束：{}", path);
        return true;
    }
    SPDLOG_DEBUG("结束RAG任务：{}", path);
    iterator->second->stop = true;
    return true;
}

bool lifuren::RAGService::removeRAGTask(const std::string& path) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto iterator = this->tasks.find(path);
    if(iterator == this->tasks.end()) {
        SPDLOG_DEBUG("RAG任务已经移除：{}", path);
        return true;
    }
    if(!iterator->second->stop && !iterator->second->finish) {
        SPDLOG_DEBUG("移除RAG任务失败：{} - {} - {}", path, iterator->second->stop, iterator->second->finish);
        return false;
    }
    SPDLOG_DEBUG("移除RAG任务：{}", path);
    return this->tasks.erase(path) > 0;
}

size_t lifuren::RAGService::taskCount() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return this->tasks.size();
}
