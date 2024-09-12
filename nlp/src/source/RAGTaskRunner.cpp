#include "lifuren/RAG.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"

#include "nlohmann/json.hpp"

lifuren::RAGTaskRunner::RAGTaskRunner(lifuren::RAGTask task) : task(task) {
    this->ragClient = lifuren::RAGClient::getRAGClient(task.type, task.path, task.embedding);
    if(!this->ragClient) {
        this->stop   = true;
        this->finish = true;
        return;
    }
    this->ragClient->loadIndex();
    this->id = this->ragClient->id;
}

lifuren::RAGTaskRunner::~RAGTaskRunner() {
    SPDLOG_DEBUG("RAG任务执行器析构：{}", this->task.path);
}

bool lifuren::RAGTaskRunner::startExecute() {
    std::lock_guard<std::mutex> lock(this->mutex);
    if(this->thread) {
        SPDLOG_DEBUG("RAG任务已经开始：{}", this->task.path);
        return true;
    }
    if(!this->ragClient) {
        SPDLOG_WARN("RAG任务没有终端：{}", this->task.path);
        return false;
    }
    this->thread = std::make_unique<std::thread>([this]() {
        try {
            SPDLOG_INFO("开始执行RAG任务：{}", this->task.path);
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
        auto& ragService = lifuren::RAGService::getInstance();
        ragService.removeRAGTask(this->task.path);
    });
    this->thread->detach();
    return true;
}

bool lifuren::RAGTaskRunner::deleteRAG() {
    if(!this->ragClient) {
        return false;
    }
    return this->ragClient->deleteRAG();
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

bool lifuren::RAGTaskRunner::execute() {
    if(!this->ragClient) {
        SPDLOG_WARN("RAG任务没有就绪：{}", this->task.path);
        return false;
    }
    if(!lifuren::files::exists(this->task.path)) {
        SPDLOG_WARN("RAG任务目录无效：{}", this->task.path);
        return false;
    }
    std::vector<std::string> vector;
    lifuren::files::listFiles(vector, this->task.path, { ".json" });
    this->fileCount = vector.size();
    SPDLOG_DEBUG("RAG任务文件总量：{} - {}", this->task.path, this->fileCount);
    for(const auto& path : vector) {
        if(this->stop) {
            break;
        }
        if(!lifuren::files::isFile(path)) {
            SPDLOG_DEBUG("RAG任务跳过其他文件：{}", path);
            continue;
        }
        if(this->ragClient->doneFileContains(path)) {
            SPDLOG_DEBUG("RAG任务跳过已经处理过的文件：{}", path);
            continue;
        }
        this->ragClient->doneFileEmplace(path);
        SPDLOG_DEBUG("RAG任务处理文件：{}", path);
        std::string&& content = lifuren::files::loadFile(path);
        if(content.empty()) {
            continue;
        }
        nlohmann::json json = nlohmann::json::parse(content);
        for(const auto& poetry : json) {
            if(this->stop) {
                break;
            }
            if(poetry.empty()) {
                continue;
            }
            auto title      = poetry.find("title");
            auto rhythm     = poetry.find("rhythm");
            auto paragraphs = poetry.find("paragraphs");
            std::string chunk;
            if(title != poetry.end()) {
                chunk += title->get<std::string>();
                chunk += '\n';
            }
            if(rhythm != poetry.end()) {
                chunk += rhythm->get<std::string>();
                chunk += '\n';
            }
            if(paragraphs != poetry.end()) {
                std::for_each(paragraphs->begin(), paragraphs->end(), [&chunk](const auto& paragraph) {
                    chunk += paragraph;
                    chunk += '\n';
                });
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
