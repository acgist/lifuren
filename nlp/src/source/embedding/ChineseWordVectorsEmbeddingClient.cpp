/**
 * https://github.com/Embedding/Chinese-Word-Vectors
 */
#include "lifuren/EmbeddingClient.hpp"

#include <map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>

#include "spdlog/spdlog.h"

static std::mutex mutex;
static std::atomic<int> share_count(0);
static std::unordered_map<std::string, std::vector<float>> vectors;

static void initVectors();
static void loadVectors(const std::string& path);

lifuren::ChineseWordVectorsEmbeddingClient::ChineseWordVectorsEmbeddingClient() : EmbeddingClient() {
    ++share_count;
}

lifuren::ChineseWordVectorsEmbeddingClient::~ChineseWordVectorsEmbeddingClient() {
    if(--share_count <= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        SPDLOG_DEBUG("ChineseWordVectors没有引用释放缓存内容");
        vectors.clear();
    }
}

std::vector<float> lifuren::ChineseWordVectorsEmbeddingClient::getVector(const std::string& prompt) const {
    initVectors();
    auto iterator = vectors.find(prompt);
    if(iterator == vectors.end()) {
        return {};
    }
    return iterator->second;
}

size_t lifuren::ChineseWordVectorsEmbeddingClient::getDims() const {
    return 300;
}

static void initVectors() {
    if(!vectors.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if(!vectors.empty()) {
        return;
    }
    auto aTime = std::chrono::system_clock::now();
    loadVectors(lifuren::config::CONFIG.chineseWordVectors.path);
    auto zTime = std::chrono::system_clock::now();
    SPDLOG_DEBUG("加载ChineseWordVectors耗时：{}毫秒", std::chrono::duration_cast<std::chrono::milliseconds>(zTime - aTime).count());
}

// TODO: 性能优化到一秒内
static void loadVectors(const std::string& path) {
    if(path.empty()) {
        SPDLOG_WARN("加载ChineseWordVectors失败（没有配置文件）：{}", path);
        return;
    }
    std::ifstream input;
    input.open(path, std::ios::in);
    if(!input.is_open()) {
        SPDLOG_WARN("加载ChineseWordVectors失败（文件打开失败）：{}", path);
        input.close();
        return;
    }
    SPDLOG_DEBUG("加载ChineseWordVectors：{}", path);
    char * beg { nullptr };
    char * pos { nullptr };
    char * lend{ nullptr };
    char * bend{ nullptr };
    size_t dims{ 0 };
    const size_t size = std::filesystem::file_size(std::filesystem::u8path(path));
    std::vector<char> data(size);
    char *buffer = data.data();
    input.read(buffer, size);
    bend = buffer + input.gcount();
    lend = std::find(buffer, bend, '\n');
    pos  = std::find(buffer, lend, ' ');
    dims = std::strtod(pos, &pos);
    pos  = lend + 1;
    beg  = pos;
    if(lend == bend || dims == 0) {
        SPDLOG_WARN("加载ChineseWordVectors失败（数据格式错误）：{}", path);
        input.close();
        return;
    }
    std::string word;
    std::vector<float> vector;
    // 使用临时变量接收最后赋值防止重入问题
    std::unordered_map<std::string, std::vector<float>> copy;
    while(true) {
        vector.reserve(dims);
        lend = std::find(beg, bend, '\n');
        pos  = std::find(beg, lend, ' ');
        word = std::string(beg, pos);
        if(word.empty()) {
            break;
        }
        ++pos;
        while(pos < lend) {
            vector.emplace_back(std::strtof(pos, &pos));
            ++pos;
        }
        copy.emplace(word, std::move(vector));
        if(lend < bend) {
            pos = lend + 1;
            beg = pos;
        } else {
            break;
        }
    }
    SPDLOG_DEBUG("加载ChineseWordVectors完成：{} - {}", copy.size(), dims);
    vectors.swap(copy);
    input.close();
}
