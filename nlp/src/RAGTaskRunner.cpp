#include "lifuren/RAG.hpp"

#include <fstream>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/String.hpp"
#include "lifuren/poetry/PoetryDataset.hpp"

// 诗词嵌入
static bool embedding(const nlohmann::json& json, std::ofstream& stream, lifuren::RAGClient* client);

lifuren::RAGTaskRunner::RAGTaskRunner(
    lifuren::RAGTask task
) :
    task(task),
    ragClient(lifuren::RAGClient::getClient(task))
{
    if(!this->ragClient) {
        this->stop   = true;
        this->finish = true;
        SPDLOG_WARN("RAG任务没有就绪：{}", task.path);
        return;
    }
    if(this->ragClient->loadIndex()) {
        this->id = this->ragClient->id;
        SPDLOG_DEBUG("RAG任务准备就绪：{} - {}", this->id, task.path);
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
        // 移除RAG任务
        lifuren::RAGService::getInstance().removeRAGTask(this->task.path);
        // 完成回调
        if(this->percentCallback) {
            this->percentCallback(this->percent(), true);
        }
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
    // 打开Embedding文件
    const std::filesystem::path embeddingPath = lifuren::file::join({ this->task.path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::EMBEDDING_MODEL_FILE });
    lifuren::file::createFolder(embeddingPath.parent_path());
    std::ofstream stream;
    stream.open(embeddingPath, std::ios_base::out | std::ios_base::app | std::ios_base::binary);
    if(!stream.is_open()) {
        SPDLOG_DEBUG("打开Embedding文件失败：{}", embeddingPath.string());
        stream.close();
        return false;
    }
    SPDLOG_DEBUG("保存词嵌入文件：{}", embeddingPath.string());
    std::vector<std::string> files;
    lifuren::file::listFile(files, this->task.path, { ".json" });
    this->fileCount = files.size();
    SPDLOG_DEBUG("RAG任务文件总量：{} - {}", this->task.path, this->fileCount);
    size_t count = 0; // 处理成功诗词总数
    size_t total = 0; // 累计读取诗词总数
    for(const auto& file : files) {
        if(this->stop) {
            break;
        }
        if(!lifuren::file::isFile(file)) {
            SPDLOG_DEBUG("RAG任务跳过其他文件：{}", file);
            continue;
        }
        if(this->ragClient->doneFileEmplace(file)) {
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
            if(this->stop) {
                // 注意这里直接跳出提前结束导致数据缺失
                // 如果需要保证数据完整建议不要直接跳出
                break;
            }
            if(poetry.empty() || !poetry.is_object()) {
                SPDLOG_WARN("RAG任务文件格式错误：{}", file);
                continue;
            }
            if(embedding(poetry, stream, this->ragClient.get())) {
                ++count;
            } else {
                // SPDLOG_WARN("RAG任务嵌入失败：{}", file);
            }
            if(total % 100 == 0) {
                SPDLOG_DEBUG("当前处理诗词数量：{} / {}", count, total);
            }
        }
        ++this->doneFileCount;
        if(this->percentCallback) {
            this->percentCallback(this->percent(), false);
        }
    }
    SPDLOG_DEBUG("累计处理诗词数量：{} / {}", count, total);
    lifuren::dataset::poetry::writeEnd(stream, lifuren::dataset::poetry::END_OF_DATASET);
    stream.close();
    return true;
}

static bool embedding(const nlohmann::json& json, std::ofstream& stream, lifuren::RAGClient* ragClient) {
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
    return true;
}
