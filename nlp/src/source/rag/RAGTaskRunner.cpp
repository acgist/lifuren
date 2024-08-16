#include "lifuren/RAG.hpp"

#include <fstream>
#include <filesystem>

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"
#include "lifuren/Lifuren.hpp"

lifuren::RAGTaskRunner::RAGTaskRunner(lifuren::RAGTask task) : task(task) {
    this->initIndex();
    this->ragClient    = lifuren::RAGClient::getRAGClient(task.rag, this->id, task.path, task.embedding);
    this->chunkService = std::make_unique<lifuren::ChunkService>(this->task.chunk);
    this->thread = std::make_unique<std::thread>([this]() {
        try {
            if(this->execute()) {
                SPDLOG_INFO("RAG任务执行完成：{}", this->task.path);
            } else {
                SPDLOG_INFO("RAG任务执行失败：{}", this->task.path);
            }
            this->saveIndex();
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
    // this->saveIndex();
}

void lifuren::RAGTaskRunner::initIndex() {
    std::filesystem::path path = this->task.path;
    path = path / "index" / "lifuren.index";
    if(std::filesystem::exists(path)) {
        std::ifstream stream;
        stream.open(path, std::ios_base::in);
        if(!stream.is_open()) {
            stream.close();
            this->id = lifuren::uuid();
            return;
        }
        std::string line;
        while(std::getline(stream, line)) {
            if(line.empty()) {
                continue;
            }
            if(this->id == 0L) {
                this->id = std::atoll(line.c_str());
            } else {
                this->doneFile.emplace(line);
            }
        }
        stream.close();
    } else {
        this->id = lifuren::uuid();
    }
}

void lifuren::RAGTaskRunner::saveIndex() {
    std::filesystem::path path = this->task.path;
    path = path / "index" / "lifuren.index";
    lifuren::files::createParent(path.string());
    std::ofstream stream;
    stream.open(path, std::ios_base::out | std::ios_base::trunc);
    if(!stream.is_open()) {
        stream.close();
        return;
    }
    stream << this->id << '\n';
    for(auto& line : this->doneFile) {
        stream << line << '\n';
    }
    stream.close();
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
        if(this->doneFile.contains(path)) {
            SPDLOG_DEBUG("RAG任务跳过已经处理过的任务：{}", path);
            continue;
        }
        if(!std::filesystem::is_regular_file(path)) {
            SPDLOG_DEBUG("RAG任务跳过其他文件：{}", path);
            continue;
        }
        this->doneFile.emplace(path);
        SPDLOG_DEBUG("RAG任务处理文件：{}", path);
        auto&& chunks = this->chunkService->chunk(path);
        for(auto& chunk : chunks) {
            if(chunk.empty()) {
                continue;
            }
            this->ragClient->index(chunk);
        }
        ++this->doneFileCount;
    }
    return true;
}

float lifuren::RAGTaskRunner::percent() {
    if(this->fileCount <= 0) {
        return 0.0F;
    }
    return static_cast<float>(this->doneFileCount) / this->fileCount;
}
