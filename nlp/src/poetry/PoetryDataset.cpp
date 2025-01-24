#include "lifuren/poetry/PoetryDataset.hpp"

#include <algorithm>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/String.hpp"
#include "lifuren/RAGClient.hpp"
#include "lifuren/poetry/Poetry.hpp"
#include "lifuren/EmbeddingClient.hpp"

static std::mutex embedding_mutex; // rag和pepper公用

// 诗词嵌入
static bool embedding(const nlohmann::json& json, std::ofstream& stream, lifuren::RAGClient* client, std::atomic_int& wCount);

lifuren::poetry::Poetry& lifuren::poetry::Poetry::preproccess() {
    if(this->title.empty() && !this->rhythmic.empty()) {
        this->title = this->rhythmic;
    }
    if(!this->title.empty() && this->rhythmic.empty()) {
        this->rhythmic = this->title;
    }
    const std::string content = std::move(lifuren::string::join(this->paragraphs, ""));
    this->simpleParagraphs    = std::move(lifuren::string::split(content, lifuren::poetry::POETRY_SIMPLE));
    this->segment             = std::move(lifuren::string::join(this->paragraphs, "\n"));
    this->simpleSegment       = std::move(lifuren::string::join(this->simpleParagraphs, "\n"));
    return *this;
}

bool lifuren::poetry::Poetry::matchRhythm() {
    if(this->rhythmPtr) {
        return true;
    }
    std::vector<uint32_t> segmentRule(this->simpleParagraphs.size());
    std::transform(this->simpleParagraphs.begin(), this->simpleParagraphs.end(), segmentRule.begin(), [](const auto& v) -> uint32_t {
        return static_cast<uint32_t>(lifuren::string::length(v));
    });
          auto beg = lifuren::config::RHYTHM.begin();
    const auto end = lifuren::config::RHYTHM.end();
    for(; beg != end; ++beg) {
        const lifuren::config::Rhythm& rhythmRef = beg->second;
        if(
            rhythmRef.segmentSize == static_cast<int>(this->simpleParagraphs.size()) &&
            rhythmRef.segmentRule == segmentRule
        ) {
            this->rhythmPtr = &rhythmRef;
            if(this->title.empty()) {
                this->title = rhythmRef.title;
            }
            if(this->rhythmic.empty()) {
                this->rhythmic = rhythmRef.rhythm;
            }
            break;
        }
    }
    return this->rhythmPtr;
}

bool lifuren::poetry::Poetry::participle() {
    if(this->rhythmPtr == nullptr) {
        return false;
    }
    uint32_t pos = 0;
    uint32_t len = 0;
    std::string word;
    const std::vector<uint32_t>& participleRuleRef = this->rhythmPtr->participleRule;
          auto beg = participleRuleRef.begin();
    const auto end = participleRuleRef.end();
          auto paragraphsBeg = this->simpleParagraphs.begin();
    const auto paragraphsEnd = this->simpleParagraphs.end();
    for(; beg != end; ++beg) {
        word = std::move(lifuren::string::substr(paragraphsBeg->c_str(), pos, *beg));
        pos += *beg;
        len += word.length();
        this->participleParagraphs.push_back(word);
        if(this->participleSegment.empty()) {
            this->participleSegment = word;
        } else {
            this->participleSegment = this->participleSegment + word;
        }
        if(len >= paragraphsBeg->length()) {
            pos = 0;
            len = 0;
            ++paragraphsBeg;
            if(paragraphsBeg == paragraphsEnd) {
                break;
            }
            this->participleSegment += "\n";
        } else {
            this->participleSegment += " ";
        }
    }
    return true;
}

bool lifuren::poetry::Poetry::operator==(const lifuren::poetry::Poetry& poetry) const {
    if(this == &poetry) {
        return true;
    }
    // 内容相同即可
    return this->paragraphs == poetry.paragraphs;
}

bool lifuren::poetry::ragEmbedding(const std::shared_ptr<lifuren::RAGClient> ragClient, const std::string& path, const std::string& dataset, std::ofstream& stream, lifuren::thread::ThreadPool& pool) {
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

bool lifuren::poetry::pepperEmbedding(const std::string& path, const std::string& dataset, std::ofstream& stream, lifuren::thread::ThreadPool& pool) {
    // 分词
    int64_t fSize = 0; // 文件数量
    int64_t wSize = 0; // 分词数量
    int64_t count = 0; // 格律数量
    int64_t total = 0; // 诗词数量
    std::set<std::string>    words;
    std::vector<std::string> files;
    lifuren::file::listFile(files, dataset, { ".json" });
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
                    ++wSize;
                    words.insert(word);
                }
            } else {
                // 匹配失败
            }
            if(total % 1000 == 0) {
                SPDLOG_DEBUG("词语数量：{} / {} / {} / {}", fSize, wSize, count, total);
            }
        }
    }
    SPDLOG_DEBUG("词语数量：{} / {} / {} / {}", fSize, wSize, count, total);
    // 嵌入
    auto iter = words.begin();
    std::shared_ptr<lifuren::EmbeddingClient> embeddingClient = std::move(lifuren::EmbeddingClient::getClient(path, "ollama"));
    const int batch_size = 512;
    static size_t wDoneSize = 0;
    wDoneSize = 0;
    for(int i = 0; i < wSize; i += batch_size) {
        std::vector<std::string> vector;
        for(int j = 0; j < batch_size && iter != words.end(); ++j, ++iter) {
            vector.push_back(std::move(*iter));
        }
        pool.submit([&stream, wSize, words = std::move(vector), embeddingClient]() {
            for(const auto& word : words) {
                SPDLOG_DEBUG("处理词语：{}", word);
                auto x = std::move(embeddingClient->getVector(word));
                {
                    std::lock_guard<std::mutex> lock(embedding_mutex);
                    ++wDoneSize;
                    size_t iSize = word.size();
                    stream.write(reinterpret_cast<char*>(&iSize), sizeof(size_t));
                    stream.write(word.data(), word.size());
                    size_t xSize = x.size();
                    stream.write(reinterpret_cast<char*>(&xSize), sizeof(size_t));
                    stream.write(reinterpret_cast<char*>(x.data()), xSize * sizeof(float));
                }
                if(wDoneSize % 100 == 0) {
                    SPDLOG_DEBUG("处理词汇数量：{} / {}", wDoneSize, wSize);
                }
            }
        });
    }
    return true;
}

bool lifuren::poetry::read(std::ifstream& stream, std::vector<std::vector<float>>& vector) {
    short size{ 0 };
    // short不能使用>>读取
    while(stream.read(reinterpret_cast<char*>(&size), sizeof(size)) && size >= 0) {
        std::vector<float> v;
        v.resize(size);
        stream.read(reinterpret_cast<char*>(v.data()), sizeof(float) * size);
        vector.push_back(std::move(v));
    }
    return stream.eof() || stream.fail();
}

void lifuren::poetry::write(std::ofstream& stream, const std::vector<std::vector<float>>& vector) {
    std::for_each(vector.begin(), vector.end(), [&stream](const std::vector<float>& v) {
        const short size = static_cast<short>(v.size());
        stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
        stream.write(reinterpret_cast<const char*>(v.data()), sizeof(float) * size);
    });
}

bool lifuren::poetry::fillRhythm(const int& dims, std::vector<std::vector<float>>& vector, const lifuren::config::Rhythm* rhythm) {
    if(rhythm == nullptr) {
        return false;
    }
    // TODO: 平仄维度
    if(dims < rhythm->fontSize) {
        SPDLOG_WARN("诗词长度超过向量维度：{} - {} - {}", rhythm->rhythm, rhythm->fontSize, dims);
        return false;
    }
    std::vector<float> segmentRule;
    segmentRule.resize(dims, 0.0F);
    std::for_each(rhythm->segmentRule.begin(), rhythm->segmentRule.end(), [pos = 0, &segmentRule](const auto& index) mutable {
        segmentRule[pos += index] = 1.0F;
    });
    std::vector<float> participleRule;
    participleRule.resize(dims, 0.0F);
    std::for_each(rhythm->participleRule.begin(), rhythm->participleRule.end(), [pos = 0, &participleRule](const auto& index) mutable {
        participleRule[pos += index] = 1.0F;
    });
    vector.push_back(std::move(segmentRule));
    vector.push_back(std::move(participleRule));
    return true;
}

lifuren::dataset::FileDatasetLoader lifuren::poetry::loadFileDatasetLoader(
    const size_t& batch_size,
    const std::string& path
) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        [](const std::string& file, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) {
            std::ifstream stream;
            stream.open(file, std::ios_base::in | std::ios_base::binary);
            if(!stream.is_open()) {
                SPDLOG_WARN("诗词嵌入文件打开失败：{}", file);
                stream.close();
                return;
            }
            std::vector<torch::Tensor> data;
            std::vector<std::vector<float>> vector;
            while(!lifuren::poetry::read(stream, vector)) {
                for(auto& v : vector) {
                    data.push_back(torch::from_blob(v.data(), { static_cast<int>(v.size()) }, torch::kFloat32).to(device).clone());
                }
                int index = 0;
                auto beg = data.begin();
                auto end = data.end();
                auto segmentRule    = beg++;
                auto participleRule = beg++;
                const int sequenceLength = lifuren::config::CONFIG.poetry.length;
                for(; beg + sequenceLength != end; ++beg, ++index) {
                    labels.push_back(*(beg + sequenceLength));
                    features.push_back(cat(segmentRule, participleRule, beg, index, device));
                }
                // EOF
                labels.push_back(torch::zeros({ beg->sizes()[0] }).to(device).clone());
                features.push_back(cat(segmentRule, participleRule, beg, index, device));
                data.clear();
                vector.clear();
            }
            stream.close();
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

bool lifuren::poetry::datasetPepperPreprocessing(const std::string& path) {
    return lifuren::dataset::allDatasetPreprocessing(path, lifuren::config::PEPPER_MODEL_FILE, &lifuren::poetry::pepperEmbedding, true);
}

bool lifuren::poetry::datasetPoetryPreprocessing(const std::string& path, const std::string& rag_type, const std::string& embedding_type) {
    std::shared_ptr<lifuren::RAGClient> client = std::move(lifuren::RAGClient::getClient(rag_type, path, embedding_type));
    // std::function<bool(const std::string&, const std::string&, std::ofstream&, lifuren::thread::ThreadPool&)>
    auto embedding = std::bind(&lifuren::poetry::ragEmbedding, client, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
    return lifuren::dataset::allDatasetPreprocessing(path, lifuren::config::EMBEDDING_MODEL_FILE, embedding);
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
