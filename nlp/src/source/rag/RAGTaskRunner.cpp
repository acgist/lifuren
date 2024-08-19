#include "lifuren/RAG.hpp"

#include <filesystem>

#include "spdlog/spdlog.h"

lifuren::RAGTaskRunner::RAGTaskRunner(lifuren::RAGTask task) : task(task) {
    this->ragClient = lifuren::RAGClient::getRAGClient(task.rag, task.path, task.embedding);
    this->ragClient->loadIndex();
    this->chunkService = std::make_unique<lifuren::ChunkService>(this->task.chunk);
    this->thread = std::make_unique<std::thread>([this]() {
        try {
            if(this->execute()) {
                SPDLOG_INFO("RAG任务执行完成：{}", this->task.path);
            } else {
                SPDLOG_INFO("RAG任务执行失败：{}", this->task.path);
            }
            this->ragClient->saveIndex();
            this->finish = true;
            this->doneFileCount = this->fileCount;
        } catch(const std::exception& e) {
            SPDLOG_ERROR("执行RAG任务异常：{} - {}", this->task.path, e.what());
        } catch(...) {
            SPDLOG_ERROR("执行RAG任务异常：{}", this->task.path);
        }
    });
    this->thread->detach();
}

lifuren::RAGTaskRunner::~RAGTaskRunner() {
    SPDLOG_DEBUG("RAG任务析构：{}", this->task.path);
}

bool lifuren::RAGTaskRunner::execute() {
    if(!std::filesystem::exists(this->task.path)) {
        SPDLOG_WARN("RAG任务目录无效：{}", this->task.path);
        return false;
    }
    if(!this->ragClient || !this->chunkService) {
        SPDLOG_WARN("RAG任务没有就绪：{}", this->task.path);
        return false;
    }
    auto iterator = std::filesystem::directory_iterator(this->task.path);
    std::list<std::string> list;
    for(const auto& entry : iterator) {
        list.push_back(entry.path().string());
    }
    this->fileCount = list.size();
    SPDLOG_DEBUG("RAG任务文件总量：{}", this->fileCount);
    for(auto& path : list) {
        if(this->stop) {
            break;
        }
        if(this->ragClient->doneFileContains(path)) {
            SPDLOG_DEBUG("RAG任务跳过已经处理过的任务：{}", path);
            continue;
        }
        if(!std::filesystem::is_regular_file(path)) {
            SPDLOG_DEBUG("RAG任务跳过其他文件：{}", path);
            continue;
        }
        this->ragClient->doneFileEmplace(path);
        SPDLOG_DEBUG("RAG任务处理文件：{}", path);
        auto&& chunks = this->chunkService->chunk(path);
        for(auto& chunk : chunks) {
            if(chunk.empty()) {
                continue;
            }
            this->ragClient->index(chunk);
        }
        ++this->doneFileCount;
        if(this->percentCallback) {
            this->percentCallback(this->percent(), false);
        }
    }
    if(this->percentCallback) {
        this->percentCallback(this->percent(), true);
    }
    return true;
}

float lifuren::RAGTaskRunner::percent() {
    if(this->fileCount <= 0) {
        return 0.0F;
    }
    return static_cast<float>(this->doneFileCount) / this->fileCount;
}

void lifuren::RAGTaskRunner::registerCallback(std::function<void(float, bool)> percentCallback) {
    this->percentCallback = percentCallback;
}

void lifuren::RAGTaskRunner::unregisterCallback() {
    this->percentCallback = nullptr;
}
