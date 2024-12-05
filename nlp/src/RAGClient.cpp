#include "lifuren/RAG.hpp"

#include <mutex>
#include <atomic>
#include <fstream>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/String.hpp"
#include "lifuren/Lifuren.hpp"
#include "lifuren/poetry/PoetryDataset.hpp"

// 避免单次重复索引
static std::set<std::string> promptCache;
// 索引计数
static std::atomic<int> share_count(0);
// 锁
static std::mutex mutex;

// 诗词嵌入
static bool embedding(const nlohmann::json& json, std::ofstream& stream, lifuren::RAGClient* client, size_t& wCount);

lifuren::RAGClient::RAGClient(
    const std::string& path,
    const std::string& embedding
) :
    path(path),
    embeddingClient(lifuren::EmbeddingClient::getClient(embedding))
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
    SPDLOG_DEBUG("保存索引文件：{}", markPath.string());
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

bool lifuren::RAGClient::donePromptEmplace(const std::string& prompt) {
    std::lock_guard<std::mutex> lock(mutex);
    auto iterator = promptCache.find(prompt);
    if(iterator == promptCache.end()) {
        promptCache.insert(prompt);
        return false;
    }
    return true;
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

bool lifuren::RAGClient::rag(const std::string& rag, const std::string& path, const std::string& embedding) {
    auto client = lifuren::RAGClient::getClient(rag, path, embedding);
    if(!client) {
        return false;
    }
    client->loadIndex();
    SPDLOG_INFO("开始执行RAG任务：{}", path);
    // 打开Embedding文件
    const std::filesystem::path embeddingPath = lifuren::file::join({ path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::EMBEDDING_MODEL_FILE });
    lifuren::file::createFolder(embeddingPath.parent_path());
    std::ofstream stream;
    stream.open(embeddingPath, std::ios_base::out | std::ios_base::app | std::ios_base::binary);
    if(!stream.is_open()) {
        SPDLOG_DEBUG("打开Embedding文件失败：{}", embeddingPath.string());
        stream.close();
        return false;
    }
    // 文件总数
    uint32_t fileCount = 0;
    // 处理文件总数
    uint32_t doneFileCount = 0;
    SPDLOG_DEBUG("保存词嵌入文件：{}", embeddingPath.string());
    std::vector<std::string> files;
    lifuren::file::listFile(files, path, { ".json" });
    fileCount = files.size();
    SPDLOG_DEBUG("RAG任务文件总量：{} - {}", path, fileCount);
    size_t count = 0; // 处理成功诗词总数
    size_t total = 0; // 累计读取诗词总数
    size_t wCount = 0; // 累计处理词语总数
    for(const auto& file : files) {
        // if(stop) {
        //     break;
        // }
        if(!lifuren::file::isFile(file)) {
            SPDLOG_DEBUG("RAG任务跳过其他文件：{}", file);
            continue;
        }
        if(client->doneFileEmplace(file)) {
            SPDLOG_DEBUG("RAG任务跳过已经处理过的文件：{}", file);
            continue;
        }
        SPDLOG_DEBUG("RAG任务处理文件：{}", file);
        const std::string content = std::move(lifuren::file::loadFile(file));
        if(content.empty()) {
            SPDLOG_WARN("RAG任务文件内容为空：{}", file);
            continue;
        }
        const nlohmann::json poetries = std::move(nlohmann::json::parse(content));
        for(const auto& poetry : poetries) {
            ++total;
            // if(stop) {
            //     // 注意这里直接跳出提前结束导致数据缺失
            //     // 如果需要保证数据完整建议不要直接跳出
            //     break;
            // }
            if(poetry.empty() || !poetry.is_object()) {
                SPDLOG_WARN("RAG任务文件格式错误：{}", file);
                continue;
            }
            if(::embedding(poetry, stream, client.get(), wCount)) {
                ++count;
            } else {
                // SPDLOG_WARN("RAG任务嵌入失败：{}", file);
            }
            if(total % 100 == 0) {
                SPDLOG_DEBUG("当前处理诗词数量：{} / {}", count, total);
            }
        }
        doneFileCount;
    }
    SPDLOG_DEBUG("累计处理诗词数量：{} / {} / {}", count, total, wCount);
    lifuren::dataset::poetry::writeEnd(stream, lifuren::dataset::poetry::END_OF_DATASET);
    stream.close();
    client->saveIndex();
    SPDLOG_INFO("RAG任务执行完成：{}", path);
    return true;
}

static bool embedding(const nlohmann::json& json, std::ofstream& stream, lifuren::RAGClient* ragClient, size_t& wCount) {
    const std::string& participle = lifuren::config::CONFIG.embedding.participle;
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
    if(lifuren::dataset::poetry::fillRhythm(ragClient->getDims(), ret, poetry.rhythmPtr)) {
        // 分段字数
        // 分词字数
        padding = 2;
    }
    ret.resize(words.size() + padding);
    std::transform(words.begin(), words.end(), ret.begin() + padding, [&ragClient](const auto& word) {
        return ragClient->index(word);
    });
    lifuren::dataset::poetry::write(stream, ret);
    wCount += ret.size();
    return true;
}
