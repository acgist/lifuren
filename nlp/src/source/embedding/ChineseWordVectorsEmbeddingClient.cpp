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
static std::unordered_map<std::string, std::vector<float>> vectors;

static void initVectors(const std::string& path);

lifuren::ChineseWordVectorsEmbeddingClient::ChineseWordVectorsEmbeddingClient() : EmbeddingClient() {
}

lifuren::ChineseWordVectorsEmbeddingClient::~ChineseWordVectorsEmbeddingClient() {
}

std::vector<float> lifuren::ChineseWordVectorsEmbeddingClient::getVector(const std::string& word) const {
    if(vectors.empty()) {
        std::lock_guard<std::mutex> lock(mutex);
        if(vectors.empty()) {
            const auto& config = lifuren::config::CONFIG.chineseWordVectors;
            if(config.path.empty()) {
                SPDLOG_WARN("加载ChineseWordVectors失败（没有配置文件）：{}", config.path);
                return {};
            }
            initVectors(config.path);
        }
    }
    auto iterator = vectors.find(word);
    if(iterator == vectors.end()) {
        return {};
    }
    return iterator->second;
}

size_t lifuren::ChineseWordVectorsEmbeddingClient::getDims() const {
    return 300;
}

bool lifuren::ChineseWordVectorsEmbeddingClient::release() {
    std::lock_guard<std::mutex> lock(mutex);
    vectors.clear();
    return lifuren::EmbeddingClient::release();
}

static void initVectors(const std::string& path) {
    std::ifstream input;
    input.open(path);
    if(!input.is_open()) {
        SPDLOG_WARN("加载ChineseWordVectors失败（文件打开失败）：{}", path);
        return;
    }
    // TODO: 优化读取
    // std::ifstream input("input.txt", std::ios::binary | std::ios::ate);
    // std::streamsize file_size = input.tellg();
    // input.seekg(0, std::ios::beg);
    // std::vector<char> buffer(file_size);
    // input.read(buffer.data(), file_size);
    // std::cout.write(buffer.data(), file_size);
    // input.close();
    SPDLOG_DEBUG("加载ChineseWordVectors：{}", path);
    size_t dims { 0 };
    size_t index{ 0 };
    std::string line;
    if(std::getline(input, line)) {
        index = line.find_first_of(" ");
        dims  = std::atoi(line.substr(index + 1).c_str());
    }
    char *pos{ nullptr };
    char *old{ nullptr };
    // 使用临时变量接收最后赋值防止重入问题
    static std::unordered_map<std::string, std::vector<float>> copy;
    while(std::getline(input, line)) {
        if(line.empty()) {
            break;
        }
        std::string word;
        std::vector<float> vector;
        vector.reserve(dims);
        index = line.find_first_of(" ");
        word  = line.substr(0, index);
        pos   = &line.at(index + 1);
        while(*pos) {
            old = pos;
            vector.emplace_back(std::strtof(pos, &pos));
            if(pos == old) {
                break;
            } else {
                ++pos;
            }
        }
        copy.emplace(word, std::move(vector));
    }
    SPDLOG_DEBUG("加载ChineseWordVectors完成：{} - {}", copy.size(), dims);
    vectors = std::move(copy);
    input.close();
}
