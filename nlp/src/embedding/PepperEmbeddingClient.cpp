#include "lifuren/EmbeddingClient.hpp"

#include <map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/poetry/Poetry.hpp"

// 读取锁：防止多线程重复读取文件
static std::mutex mutex;
// 共享数量：没有引用时释放内存
static std::atomic<int> share_count(0);
// 嵌入向量
static std::unordered_map<std::string, std::vector<float>> vectors;

// 加载嵌入向量
static void initVectors(const std::string& pepperPath);
// 加载嵌入向量
static void loadVectors(const std::string& path);

lifuren::PepperEmbeddingClient::PepperEmbeddingClient(const std::string& path) : EmbeddingClient(path) {
    ++share_count;
}

lifuren::PepperEmbeddingClient::~PepperEmbeddingClient() {
    if(--share_count <= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        SPDLOG_DEBUG("没有引用释放嵌入向量");
        vectors.clear();
    }
}

std::vector<float> lifuren::PepperEmbeddingClient::getVector(const std::string& prompt) const {
    auto pepperPath = lifuren::file::join({ this->path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::PEPPER_MODEL_FILE });
    initVectors(pepperPath.string());
    auto iterator = vectors.find(prompt);
    if(iterator == vectors.end()) {
        SPDLOG_WARN("没有匹配嵌入内容：{}", prompt);
        return {};
    }
    return iterator->second;
}

size_t lifuren::PepperEmbeddingClient::getDims() const {
    return lifuren::config::CONFIG.pepper.dims;
}

static void initVectors(const std::string& pepperPath) {
    if(!vectors.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if(!vectors.empty()) {
        return;
    }
    auto aTime = std::chrono::system_clock::now();
    loadVectors(pepperPath);
    auto zTime = std::chrono::system_clock::now();
    SPDLOG_DEBUG("加载pepper耗时：{}毫秒", std::chrono::duration_cast<std::chrono::milliseconds>(zTime - aTime).count());
}

static void loadVectors(const std::string& path) {
    if(path.empty()) {
        SPDLOG_WARN("加载pepper失败（没有配置文件）：{}", path);
        return;
    }
    std::ifstream input;
    input.open(path, std::ios_base::in | std::ios_base::binary);
    if(!input.is_open()) {
        SPDLOG_WARN("加载pepper失败（文件打开失败）：{}", path);
        input.close();
        return;
    }
    SPDLOG_DEBUG("加载pepper：{}", path);
    std::string word;
    std::vector<float> vector;
    // 使用临时变量接收最后赋值防止重入问题
    std::unordered_map<std::string, std::vector<float>> copy;
    size_t wSize;
    size_t vSize;
    while(input.read(reinterpret_cast<char*>(&wSize), sizeof(wSize))) {
        word.resize(wSize);
        input.read(word.data(), wSize);
        input.read(reinterpret_cast<char*>(&vSize), sizeof(vSize));
        vector.resize(vSize);
        input.read(reinterpret_cast<char*>(vector.data()), sizeof(float) * vSize);
        copy.emplace(std::move(word), std::move(vector));
    }
    SPDLOG_DEBUG("加载pepper完成：{} - {}", copy.size(), vSize);
    vectors.swap(copy);
    input.close();
}
