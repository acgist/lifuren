#include "lifuren/RAGClient.hpp"

#include <mutex>
#include <atomic>
#include <fstream>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/String.hpp"
#include "lifuren/poetry/PoetryDataset.hpp"

static std::mutex embedding_mutex;

// 避免单次重复索引
static std::set<std::string> promptCache;
// 索引计数
static std::atomic<int> share_count(0);
// 锁
static std::mutex mutex;

// 诗词嵌入
static bool embedding(const nlohmann::json& json, std::ofstream& stream, lifuren::RAGClient* client, std::atomic_int& wCount);

lifuren::RAGClient::RAGClient(
    const std::string& path,
    const std::string& embedding
) : path(path),
    embeddingClient(lifuren::EmbeddingClient::getClient(path, embedding))
{
    ++share_count;
}

lifuren::RAGClient::~RAGClient() {
    if(--share_count <= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        promptCache.clear();
        SPDLOG_DEBUG("没有引用清空提示索引缓存");
    }
}

size_t lifuren::RAGClient::getDims() const {
    if(this->embeddingClient) {
        return this->embeddingClient->getDims();
    } else {
        return 0;
    }
}

bool lifuren::RAGClient::donePromptEmplace(const std::string& prompt) {
    std::lock_guard<std::mutex> lock(mutex);
    auto iterator = promptCache.find(prompt);
    if(iterator == promptCache.end()) {
        promptCache.insert(prompt);
        return false;
    }
    return true;
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

bool lifuren::rag::embedding(const std::shared_ptr<lifuren::RAGClient> ragClient, const std::string& path, const std::string& dataset, std::ofstream& stream, lifuren::thread::ThreadPool& pool) {
    if(!ragClient) {
        SPDLOG_WARN("RAGClient无效：{}", dataset);
        return false;
    }
    SPDLOG_INFO("开始执行RAG任务：{}", dataset);
    std::vector<std::string> files;
    lifuren::file::listFile(files, dataset, { ".json" });
    static std::atomic_int count  = 0; // 处理成功诗词总数
    static std::atomic_int total  = 0; // 累计读取诗词总数
    static std::atomic_int wCount = 0; // 累计处理词语总数
    static std::atomic_int fileCount     = 0; // 文件总数
    static std::atomic_int doneFileCount = 0; // 处理文件总数
    count  = 0;
    total  = 0;
    wCount = 0;
    fileCount     = files.size();
    doneFileCount = 0;
    SPDLOG_DEBUG("RAG任务文件总量：{} - {}", dataset, fileCount.load());
    for(const auto& file : files) {
        pool.submit([file, &stream, ragClient]() {
            if(!lifuren::file::is_file(file)) {
                SPDLOG_DEBUG("RAG任务跳过其他文件：{}", file);
                return;
            }
            SPDLOG_DEBUG("RAG任务处理文件：{}", file);
            const std::string content = std::move(lifuren::file::loadFile(file));
            if(content.empty()) {
                SPDLOG_WARN("RAG任务文件内容为空：{}", file);
                return;
            }
            const nlohmann::json poetries = std::move(nlohmann::json::parse(content));
            for(const auto& poetry : poetries) {
                ++total;
                if(poetry.empty() || !poetry.is_object()) {
                    SPDLOG_WARN("RAG任务文件格式错误：{}", file);
                    continue;
                }
                if(::embedding(poetry, stream, ragClient.get(), wCount)) {
                    ++count;
                } else {
                    SPDLOG_WARN("RAG任务嵌入失败：{}", file);
                }
                if(total % 100 == 0) {
                    SPDLOG_DEBUG("当前处理诗词数量：{} / {}", count.load(), total.load());
                }
            }
            ++doneFileCount;
        });
    }
    return true;
}

static bool embedding(const nlohmann::json& json, std::ofstream& stream, lifuren::RAGClient* ragClient, std::atomic_int& wCount) {
    const std::string& participle = lifuren::config::CONFIG.poetry.embedding_participle;
    lifuren::poetry::Poetry poetry = json;
    poetry.preproccess();
    if(!poetry.matchRhythm()) {
        SPDLOG_WARN("诗词没有格律：{}", poetry.title);
        return false;
    }
    std::vector<std::string> words;
    if(participle == "char" || participle == "CHAR") {
        auto ret = std::move(lifuren::string::toChars(poetry.simpleSegment));
        words.insert(words.end(), ret.begin(), ret.end());
    } else if(participle == "rhythm" || participle == "RHYTHM") {
        poetry.participle();
        words.insert(words.end(), poetry.participleParagraphs.begin(), poetry.participleParagraphs.end());
    } else {
        SPDLOG_WARN("不支持的分词方式：{}", participle);
        return false;
    }
    if(words.empty()) {
        SPDLOG_WARN("诗词分词失败：{}", poetry.title);
        return false;
    }
    // TODO: 平仄维度向量=音调
    size_t padding = 0;
    std::vector<std::vector<float>> ret;
    if(lifuren::poetry::fillRhythm(ragClient->getDims(), ret, poetry.rhythmPtr)) {
        // 分段字数
        // 分词字数
        padding = 2;
    }
    ret.resize(words.size() + padding);
    std::transform(words.begin(), words.end(), ret.begin() + padding, [&ragClient](const auto& word) {
        return ragClient->index(word);
    });
    {
        std::lock_guard<std::mutex> lock(embedding_mutex);
        lifuren::poetry::write(stream, ret);
    }
    wCount += ret.size();
    return true;
}
