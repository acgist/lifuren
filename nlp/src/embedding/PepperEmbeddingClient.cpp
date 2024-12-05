/**
 * PepperEmbeddingClient.cpp
 * 
 * @author acgist
 * 
 * @version 1.0.0
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
static void initVectors();
// 加载嵌入向量
static void loadVectors(const std::string& path);

lifuren::PepperEmbeddingClient::PepperEmbeddingClient() : EmbeddingClient() {
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
    initVectors();
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

bool lifuren::PepperEmbeddingClient::embedding(const std::string& path) {
    auto pepperPath = lifuren::file::join({ path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::PEPPER_WORD_FILE });
    // 读取
    std::set<std::string> old_words;
    if(std::filesystem::exists(pepperPath)) {
        std::ifstream input;
        input.open(pepperPath, std::ios_base::in | std::ios_base::binary);
        if(!input.is_open()) {
            SPDLOG_WARN("加载pepper失败（文件打开失败）：{}", pepperPath.string());
            input.close();
            return false;
        }
        SPDLOG_DEBUG("加载pepper：{}", pepperPath.string());
        size_t wSize;
        size_t vSize;
        std::string word;
        while(input.read(reinterpret_cast<char*>(&wSize), sizeof(wSize))) {
            word.resize(wSize);
            input.read(word.data(), wSize);
            old_words.emplace(word);
            input.read(reinterpret_cast<char*>(&vSize), sizeof(vSize));
            input.seekg(vSize * sizeof(float), std::ios::cur);
        }
        input.close();
    }
    // 分词
    int64_t fSize = 0; // 文件数量
    int64_t wSize = 0; // 分词数量
    int64_t count = 0; // 诗词匹配格律数量
    int64_t total = 0; // 诗词数量
    std::set<std::string> words;
    std::vector<std::string> files;
    lifuren::file::listFile(files, lifuren::file::join({ path }).string(), { ".json" });
    for(const auto& file : files) {
        ++fSize;
        std::string json = std::move(lifuren::file::loadFile(file));
        auto poetries = nlohmann::json::parse(json);
        for(const auto& poetry : poetries) {
            ++total;
            lifuren::poetry::Poetry value = poetry;
            value.preproccess();
            if(value.matchRhythm()) {
                ++count;
                value.participle();
                for(const auto& word : value.participleParagraphs) {
                    if(old_words.contains(word)) {
                        SPDLOG_DEBUG("已经含有分词：{}", word);
                    } else {
                        ++wSize;
                        words.insert(word);
                    }
                }
            } else {
                // 匹配失败
            }
            if(total % 1000 == 0) {
                SPDLOG_DEBUG("当前数量：{} / {} / {} / {}", fSize, wSize, count, total);
            }
        }
    }
    SPDLOG_DEBUG("开始嵌入分词：{} - {}", old_words.size(), words.size());
    // 嵌入
    lifuren::file::createParent(pepperPath);
    std::ofstream output;
    output.open(pepperPath, std::ios_base::app | std::ios_base::out | std::ios_base::binary);
    if(!output.is_open()) {
        output.close();
        SPDLOG_WARN("文件打开失败：{}", pepperPath.string());
        return false;
    }
    const int batch = 10; // 线程数量
    std::mutex mutex;
    std::atomic_int countDown(batch);
    std::condition_variable condition;
    std::vector<std::string> vector;
    vector.reserve(words.size());
    vector.assign(words.begin(), words.end());
    const int batchSize = vector.size() / batch;
    auto embeddingClient = lifuren::EmbeddingClient::getClient("ollama");
    // 多线程加载
    for(int i = 0; i < batch; ++i) {
        std::thread thread([i, &mutex, &output, &vector, &countDown, &batchSize, &condition, &embeddingClient]() {
            int index = 0;
            SPDLOG_DEBUG("启动线程：{} {}", i, batch - 1);
            auto beg = vector.begin() + (i * batchSize);
            auto end = (i == batch - 1) ? vector.end() : beg + batchSize;
            for(; beg != end; ++beg) {
                auto x = std::move(embeddingClient->getVector(*beg));
                SPDLOG_DEBUG("处理词语：{} {} {} {}", i, x.size(), beg->size(), *beg);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    size_t iSize = beg->size();
                    output.write(reinterpret_cast<char*>(&iSize), sizeof(size_t));
                    output.write(beg->data(), beg->size());
                    size_t xSize = x.size();
                    output.write(reinterpret_cast<char*>(&xSize), sizeof(size_t));
                    output.write(reinterpret_cast<char*>(x.data()), xSize * sizeof(float));
                }
                if(++index % 100 == 0) {
                    SPDLOG_DEBUG("处理数量：{} - {}", i, index);
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                --countDown;
                condition.notify_all();
            }
        });
        thread.detach();
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        while(countDown != 0) {
            condition.wait(lock);
        }
    }
    output.close();
    return true;
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
    loadVectors(lifuren::config::CONFIG.pepper.path);
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
