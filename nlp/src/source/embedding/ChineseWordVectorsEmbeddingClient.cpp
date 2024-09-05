/**
 * https://github.com/Embedding/Chinese-Word-Vectors
 */
#include "lifuren/EmbeddingClient.hpp"

#include <map>
#include <mutex>
#include <thread>
#include <fstream>

#include "spdlog/spdlog.h"

static std::mutex mutex;
static std::map<std::string, std::vector<float>> vectors;

static void initVectors(const std::string& path);

lifuren::ChineseWordVectorsEmbeddingClient::ChineseWordVectorsEmbeddingClient() {
}

lifuren::ChineseWordVectorsEmbeddingClient::~ChineseWordVectorsEmbeddingClient() {
}

std::vector<float> lifuren::ChineseWordVectorsEmbeddingClient::getVector(const std::string& word) {
    std::lock_guard<std::mutex> lock(mutex);
    if(vectors.empty()) {
        const auto& config = lifuren::config::CONFIG.chineseWordVectors;
        if(config.path.empty()) {
            SPDLOG_WARN("加载ChineseWordVectors失败（没有配置文件）：{}", config.path);
            return {};
        }
        initVectors(config.path);
    }
    auto iterator = vectors.find(word);
    if(iterator == vectors.end()) {
        return {};
    }
    // 返回拷贝
    return iterator->second;
}

static void initVectors(const std::string& path) {
    std::ifstream input;
    input.open(path);
    if(!input.is_open()) {
        SPDLOG_WARN("加载ChineseWordVectors失败（文件打开失败）：{}", path);
        return;
    }
    SPDLOG_DEBUG("加载ChineseWordVectors：{}", path);
    size_t dims{ 0 };
    std::string line;
    if(std::getline(input, line)) {
        size_t index = line.find_first_of(" ");
        auto x = line.substr(index);
        dims = std::atoi(line.substr(index + 1).c_str());
    }
    while(std::getline(input, line)) {
        if(line.empty()) {
            break;
        }
        std::string word;
        std::vector<float> vector;
        vector.reserve(dims);
        size_t pos   = 0;
        size_t index = line.find_first_of(" ", pos);
        word = line.substr(pos, index);
        pos = index;
        while((index = line.find_first_of(" ", ++pos)) != std::string::npos) {
            auto x = line.substr(pos, index - pos);
            vector.emplace_back(std::atof(line.substr(pos, index - pos).c_str()));
            pos = index;
        }
        vectors.emplace(word, std::move(vector));
    }
}
