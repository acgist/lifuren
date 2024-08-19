#include "lifuren/RAG.hpp"

#include <fstream>
#include <filesystem>

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"
#include "lifuren/Lifuren.hpp"

lifuren::RAGClient::RAGClient(const std::string& path, const std::string& embedding) : path(path) {
    this->embeddingClient = lifuren::EmbeddingClient::getClient(embedding);
}

lifuren::RAGClient::~RAGClient() {
}

void lifuren::RAGClient::loadIndex() {
    std::filesystem::path path = this->path;
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

void lifuren::RAGClient::saveIndex() {
    std::filesystem::path path = this->path;
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

void lifuren::RAGClient::doneFileEmplace(const std::string& file) {
    this->doneFile.emplace(file);
}

bool lifuren::RAGClient::doneFileContains(const std::string& file) {
    return this->doneFile.contains(file);
}

std::unique_ptr<lifuren::RAGClient> lifuren::RAGClient::getRAGClient(const std::string& type, const std::string& path, const std::string& embedding) {
    if(type == "elasticsearch" || type == "ElasticSearch") {
        return std::make_unique<lifuren::ElasticSearchRAGClient>(path, embedding);
    } else {
        SPDLOG_WARN("不支持的RAGClient类型：{}", type);
    }
    return nullptr;
}
