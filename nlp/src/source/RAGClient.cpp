#include "lifuren/RAG.hpp"

#include <fstream>

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"
#include "lifuren/Lifuren.hpp"

lifuren::RAGClient::RAGClient(const std::string& path, const std::string& embedding) :
    path(path),
    embeddingClient(lifuren::EmbeddingClient::getClient(embedding))
{
}

lifuren::RAGClient::~RAGClient() {
}

size_t lifuren::RAGClient::getDims() const {
    if(this->embeddingClient) {
        return this->embeddingClient->getDims();
    } else {
        return 0;
    }
}

void lifuren::RAGClient::loadIndex() {
    const std::filesystem::path markPath = lifuren::files::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::MARK_MODEL_FILE });
    if(!std::filesystem::exists(markPath)) {
        this->id = lifuren::uuid();
    }
    std::ifstream stream;
    stream.open(markPath, std::ios_base::in);
    if(!stream.is_open()) {
        SPDLOG_WARN("RAG索引文件打开失败：{}", markPath.string());
        this->id = lifuren::uuid();
        stream.close();
        return;
    }
    std::string line;
    while(std::getline(stream, line)) {
        if(line.empty()) {
            continue;
        }
        if(this->id == 0LL) {
            this->id = std::atoll(line.c_str());
        } else {
            this->doneFile.emplace(line);
        }
    }
    stream.close();
}

void lifuren::RAGClient::saveIndex() {
    const std::filesystem::path markPath = lifuren::files::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::MARK_MODEL_FILE });
    lifuren::files::createFolder(markPath.parent_path());
    std::ofstream stream;
    stream.open(markPath, std::ios_base::out | std::ios_base::trunc);
    if(!stream.is_open()) {
        SPDLOG_WARN("RAG索引文件打开失败：{}", markPath.string());
        stream.close();
        return;
    }
    stream << this->id << '\n';
    for(const auto& line : this->doneFile) {
        stream << line << '\n';
    }
    stream.close();
}

void lifuren::RAGClient::truncateIndex() {
    SPDLOG_DEBUG("清空RAG索引：{}", this->id);
    this->doneFile.clear();
    this->saveIndex();
}

bool lifuren::RAGClient::doneFileEmplace(const std::string& file) {
    if(this->doneFile.contains(file)) {
        return true;
    }
    this->doneFile.emplace(file);
    return false;
}

std::vector<std::string> lifuren::RAGClient::search(const std::string& prompt, const int size) const {
    auto&& vector = this->embeddingClient->getVector(prompt);
    return this->search(vector, size);
}

std::unique_ptr<lifuren::RAGClient> lifuren::RAGClient::getRAGClient(const std::string& rag, const std::string& embedding, const std::string& path) {
    if(rag == "faiss" || rag == "Faiss") {
        return std::make_unique<lifuren::FaissRAGClient>(path, embedding);
    } else if(rag == "elasticsearch" || rag == "ElasticSearch") {
        return std::make_unique<lifuren::ElasticSearchRAGClient>(path, embedding);
    } else {
        SPDLOG_WARN("不支持的RAGClient类型：{}", rag);
    }
    return nullptr;
}
