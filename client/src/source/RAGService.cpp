#include "lifuren/RAGClient.hpp"

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
    auto iterator = this->tasks.find(path);
    if(iterator == this->tasks.end()) {
        return nullptr;
    }
    return iterator->second;
}

std::shared_ptr<lifuren::RAGTaskRunner> lifuren::RAGService::buildRAGTask(const std::string& path) {
    const auto& rag       = lifuren::config::CONFIG.rag;
    const auto& embedding = lifuren::config::CONFIG.embedding;
    RAGTask task{
        .rag       = rag.type,
        .path      = path,
        .embedding = embedding.type,
    };
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto iterator = this->tasks.find(task.path);
    if(iterator != this->tasks.end()) {
        SPDLOG_DEBUG("RAG任务已经添加：{}", task.path);
        return iterator->second;
    }
    SPDLOG_DEBUG("添加RAG任务：{}", task.path);
    const auto runner = std::make_shared<lifuren::RAGTaskRunner>(task);
    if(runner->id <= 0L) {
        return nullptr;
    } else {
        runner->startExecute();
        auto pair = this->tasks.emplace(task.path, runner);
        return pair.first->second;
    }
}

bool lifuren::RAGService::stopRAGTask(const std::string& path) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto iterator = this->tasks.find(path);
    if(iterator == this->tasks.end()) {
        SPDLOG_DEBUG("RAG任务已经删除：{}", path);
        return true;
    }
    SPDLOG_DEBUG("删除RAG任务：{}", path);
    iterator->second->stop = true;
    return true;
}

bool lifuren::RAGService::deleteRAGTask(const std::string& path) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto iterator = this->tasks.find(path);
    SPDLOG_DEBUG("删除RAG任务：{}", path);
    if(iterator == this->tasks.end()) {
        const auto& rag       = lifuren::config::CONFIG.rag;
        const auto& embedding = lifuren::config::CONFIG.embedding;
        RAGTask task{
            .rag       = rag.type,
            .path      = path,
            .embedding = embedding.type,
        };
        const auto runner = std::make_shared<lifuren::RAGTaskRunner>(task);
        return runner->deleteRAG();
    } else {
        iterator->second->stop = true;
        return iterator->second->deleteRAG();
    }
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