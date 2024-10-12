#include "lifuren/RAG.hpp"

#include <fstream>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Poetry.hpp"
#include "lifuren/String.hpp"
#include "lifuren/PoetryDataset.hpp"

static bool embedding(const nlohmann::json& json, std::ofstream& stream, lifuren::RAGClient* client);

lifuren::RAGTaskRunner::RAGTaskRunner(lifuren::RAGTask task) :
    task(task),
    ragClient(lifuren::RAGClient::getClient(task.rag, task.embedding, task.path))
{
    if(!this->ragClient) {
        this->stop   = true;
        this->finish = true;
        SPDLOG_WARN("RAG任务没有就绪：{}", task.path);
        return;
    }
    if(this->ragClient->loadIndex()) {
        this->id = this->ragClient->id;
    } else {
        this->stop   = true;
        this->finish = true;
        SPDLOG_WARN("RAG任务加载失败：{}", task.path);
    }
}

lifuren::RAGTaskRunner::~RAGTaskRunner() {
    SPDLOG_DEBUG("RAG任务执行器析构：{}", this->task.path);
}

bool lifuren::RAGTaskRunner::startExecute() {
    std::lock_guard<std::mutex> lock(this->mutex);
    if(this->thread) {
        SPDLOG_DEBUG("RAG任务已经开始：{}", this->task.path);
        return true;
    }
    if(!this->ragClient) {
        SPDLOG_WARN("RAG任务没有就绪：{}", this->task.path);
        return false;
    }
    this->thread = std::make_unique<std::thread>([this]() {
        try {
            SPDLOG_INFO("开始执行RAG任务：{}", this->task.path);
            if(this->execute()) {
                SPDLOG_INFO("RAG任务执行完成：{}", this->task.path);
            } else {
                SPDLOG_WARN("RAG任务执行失败：{}", this->task.path);
            }
        } catch(const std::exception& e) {
            SPDLOG_ERROR("执行RAG任务异常：{} - {}", this->task.path, e.what());
        } catch(...) {
            SPDLOG_ERROR("执行RAG任务异常：{}", this->task.path);
        }
        this->stop   = true;
        this->finish = true;
        this->ragClient->saveIndex();
        lifuren::RAGService::getInstance().removeRAGTask(this->task.path);
    });
    this->thread->detach();
    return true;
}

float lifuren::RAGTaskRunner::percent() const {
    if(this->finish) {
        return 1.0F;
    }
    if(this->fileCount <= 0) {
        return 0.0F;
    }
    return static_cast<float>(this->doneFileCount) / this->fileCount;
}

void lifuren::RAGTaskRunner::registerCallback(std::function<void(float, bool)> percentCallback) {
    this->percentCallback = percentCallback;
}

void lifuren::RAGTaskRunner::unregisterCallback() {
    this->percentCallback = nullptr;
}

bool lifuren::RAGTaskRunner::execute() {
    if(!this->ragClient) {
        SPDLOG_WARN("RAG任务没有就绪：{}", this->task.path);
        return false;
    }
    if(!lifuren::file::exists(this->task.path)) {
        SPDLOG_WARN("RAG任务目录无效：{}", this->task.path);
        return false;
    }
    std::vector<std::string> paths;
    lifuren::file::listFile(paths, this->task.path, { ".json" });
    this->fileCount = paths.size();
    SPDLOG_DEBUG("RAG任务文件总量：{} - {}", this->task.path, this->fileCount);
    const std::filesystem::path embeddingPath = lifuren::file::join({ this->task.path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::EMBEDDING_MODEL_FILE });
    lifuren::file::createFolder(embeddingPath.parent_path());
    std::ofstream stream;
    stream.open(embeddingPath, std::ios_base::out | std::ios_base::app | std::ios_base::binary);
    if(!stream.is_open()) {
        SPDLOG_DEBUG("打开Embedding文件失败：{}", embeddingPath.string());
        stream.close();
        return false;
    }
    size_t count = 0LL; // 处理成功诗词总数
    size_t total = 0LL; // 累计读取诗词总数
    for(const auto& path : paths) {
        if(this->stop) {
            break;
        }
        if(!lifuren::file::isFile(path)) {
            SPDLOG_DEBUG("RAG任务跳过其他文件：{}", path);
            continue;
        }
        if(this->ragClient->doneFileEmplace(path)) {
            SPDLOG_DEBUG("RAG任务跳过已经处理过的文件：{}", path);
            continue;
        }
        SPDLOG_DEBUG("RAG任务处理文件：{}", path);
        const std::string&& content = lifuren::file::loadFile(path);
        if(content.empty()) {
            continue;
        }
        const nlohmann::json&& poetries = nlohmann::json::parse(content);
        for(const auto& poetry : poetries) {
            ++total;
            // TODO: 是否等待处理完成
            if(this->stop) {
                break;
            }
            if(poetry.empty() || !poetry.is_object()) {
                continue;
            }
            if(embedding(poetry, stream, this->ragClient.get())) {
                ++count;
                if(count % 100 == 0) {
                    SPDLOG_DEBUG("当前处理诗词数量：{} / {}", count, total);
                }
            } else {
                // 失败
            }
        }
        ++this->doneFileCount;
        if(this->percentCallback) {
            this->percentCallback(this->percent(), false);
        }
    }
    if(this->percentCallback) {
        this->percentCallback(this->percent(), true);
    }
    SPDLOG_DEBUG("累计处理诗词数量：{} / {}", count, total);
    stream.write(reinterpret_cast<const char*>(&lifuren::dataset::poetry::END_OF_DATASET), sizeof(lifuren::dataset::poetry::END_OF_DATASET));
    stream.close();
    return true;
}

static bool embedding(const nlohmann::json& json, std::ofstream& stream, lifuren::RAGClient* ragClient) {
    const std::string& participle = lifuren::config::CONFIG.embedding.participle;
    lifuren::poetry::Poetry poetry = json;
    poetry.preproccess();
    if(!poetry.matchRhythm()) {
        SPDLOG_WARN("没有匹配格律：{}", poetry.title);
        return false;
    }
    std::vector<std::string> words;
    if(participle == "char" || participle == "CHAR") {
        auto&& ret = lifuren::string::toChars(poetry.simpleSegment);
        words.insert(words.end(), ret.begin(), ret.end());
    } else if(participle == "rhythm" || participle == "RHYTHM") {
        poetry.participle();
        words.insert(words.end(), poetry.participleParagraphs.begin(), poetry.participleParagraphs.end());
    } else {
        SPDLOG_WARN("不支持的分词方式：{}", participle);
        return false;
    }
    if(words.empty()) {
        return false;
    }
    // TODO: 平仄维度向量=音调
    std::vector<std::vector<float>> ret;
    size_t padding = 0;
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
    return true;
}
