#include "lifuren/RAG.hpp"

#include <fstream>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Lifuren.hpp"

lifuren::RAGClient::RAGClient(
    const std::string& path,
    const std::string& embedding
) :
    path(path),
    embeddingClient(lifuren::EmbeddingClient::getClient(embedding))
{
}

lifuren::RAGClient::~RAGClient() {
}

bool lifuren::RAGClient::loadIndex() {
    const std::filesystem::path markPath = lifuren::file::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::MARK_MODEL_FILE });
    if(!std::filesystem::exists(markPath)) {
        this->id = lifuren::uuid();
        return true;
    }
    std::ifstream stream;
    stream.open(markPath, std::ios_base::in);
    if(!stream.is_open()) {
        SPDLOG_WARN("RAG索引文件打开失败：{}", markPath.string());
        this->id = 0;
        stream.close();
        return false;
    }
    std::string line;
    while(std::getline(stream, line)) {
        if(line.empty()) {
            continue;
        }
        if(this->id == 0) {
            this->id = std::atoll(line.c_str());
        } else {
            this->doneFile.emplace(line);
        }
    }
    stream.close();
    return true;
}

bool lifuren::RAGClient::saveIndex() const {
    const std::filesystem::path markPath = lifuren::file::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::MARK_MODEL_FILE });
    lifuren::file::createFolder(markPath.parent_path());
    std::ofstream stream;
    stream.open(markPath, std::ios_base::out | std::ios_base::trunc);
    if(!stream.is_open()) {
        SPDLOG_WARN("RAG索引文件打开失败：{}", markPath.string());
        stream.close();
        return false;
    }
    stream << this->id << '\n';
    for(const auto& line : this->doneFile) {
        stream << line << '\n';
    }
    stream.close();
    return true;
}

bool lifuren::RAGClient::truncateIndex() {
    SPDLOG_DEBUG("清空RAG索引：{}", this->id);
    this->doneFile.clear();
    this->saveIndex();
    return true;
}

size_t lifuren::RAGClient::getDims() const {
    if(this->embeddingClient) {
        return this->embeddingClient->getDims();
    } else {
        return 0;
    }
}

bool lifuren::RAGClient::doneFileEmplace(const std::string& file) {
    if(this->doneFile.contains(file)) {
        return true;
    }
    this->doneFile.emplace(file);
    return false;
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
