#include "lifuren/RAG.hpp"

#include <mutex>

#include "spdlog/spdlog.h"

static std::mutex mutex;

lifuren::RAGService::RAGService() {
}

lifuren::RAGService::~RAGService() {
}

lifuren::RAGService& lifuren::RAGService::getInstance() {
    static lifuren::RAGService instance;
    return instance;
}

std::shared_ptr<lifuren::RAGTaskRunner> lifuren::RAGService::buildRAGTask(RAGTask task) {
    std::lock_guard<std::mutex> lock(mutex);
    auto iterator = this->tasks.find(task.path);
    if(iterator != this->tasks.end()) {
        SPDLOG_DEBUG("RAG任务已经添加：{}", task.path);
        return iterator->second;
    }
    SPDLOG_DEBUG("添加RAG任务：{}", task.path);
    auto pair = this->tasks.emplace(task.path, std::make_shared<lifuren::RAGTaskRunner>(task));
    return pair.first->second;
}

bool lifuren::RAGService::stopRAGTask(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex);
    auto iterator = this->tasks.find(path);
    if(iterator == this->tasks.end()) {
        SPDLOG_DEBUG("RAG任务已经结束：{}", path);
        return false;
    }
    SPDLOG_DEBUG("结束RAG任务：{}", path);
    iterator->second->stop = true;
    return true;
}

bool lifuren::RAGService::deleteRAGTask(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex);
    return this->tasks.erase(path) > 0;
}

size_t lifuren::RAGService::taskCount() {
    std::lock_guard<std::mutex> lock(mutex);
    return this->tasks.size();
}

float lifuren::RAGService::taskPercent(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex);
    auto iterator = this->tasks.find(path);
    if(iterator == this->tasks.end()) {
        return 0.0F;
    }
    return iterator->second->percent();
}
