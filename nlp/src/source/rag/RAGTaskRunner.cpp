#include "lifuren/RAG.hpp"

#include "spdlog/spdlog.h"

lifuren::RAGTaskRunner::RAGTaskRunner(lifuren::RAGTask task) : task(task) {
    if(task.rag == "elasticsearch") {
        this->ragClient = std::make_unique<lifuren::ElasticSearchRAGClient>(task.embedding);
    } else {

    }
    this->chunkService = std::make_unique<lifuren::ChunkService>(this->task.chunk);
    this->thread = std::make_unique<std::thread>([this]() {
        try {
            this->execute();
            this->finish = true;
        } catch(const std::exception& e) {
            SPDLOG_ERROR("执行RAG任务异常：{} - {}", this->task.path, e.what());
        } catch(...) {
            SPDLOG_ERROR("执行RAG任务异常：{}", this->task.path);
        }
    });
    this->thread->detach();
}

lifuren::RAGTaskRunner::~RAGTaskRunner() {
}

bool lifuren::RAGTaskRunner::execute() {
    if(!this->ragClient || !this->chunkService) {
        return false;
    }
    return true;
}

float lifuren::RAGTaskRunner::percent() {
    if(this->fileCount <= 0) {
        return 1.0F;
    }
    return static_cast<float>(this->successFileCount) / this->fileCount;
}
