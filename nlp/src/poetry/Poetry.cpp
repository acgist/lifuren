#include "lifuren/poetry/Poetry.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/String.hpp"
#include "lifuren/EmbeddingClient.hpp"

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

bool lifuren::poetry::embedding(const std::string& path, std::ofstream& stream, lifuren::thread::ThreadPool& pool) {
    // 分词
    int64_t fSize = 0; // 文件数量
    int64_t wSize = 0; // 分词数量
    int64_t count = 0; // 诗词匹配格律数量
    int64_t total = 0; // 诗词数量
    std::set<std::string>    words;
    std::vector<std::string> files;
    lifuren::file::listFile(files, path, { ".json" });
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
                SPDLOG_DEBUG("当前数量：{} / {} / {} / {}", fSize, wSize, count, total);
            }
        }
    }
    SPDLOG_DEBUG("开始嵌入分词：{}", words.size());
    // 嵌入
    auto iter = words.begin();
    std::shared_ptr<lifuren::EmbeddingClient> embeddingClient = std::move(lifuren::EmbeddingClient::getClient(path, "ollama"));
    const int batch_size = 512;
    for(int i = 0; i < wSize; i += batch_size) {
        std::vector<std::string> vector;
        for(int j = 0; j < batch_size && iter != words.end(); ++j, ++iter) {
            vector.push_back(std::move(*iter));
        }
        pool.enqueue([&stream, words = std::move(vector), embeddingClient]() {
            for(const auto& word : words) {
                auto x = std::move(embeddingClient->getVector(word));
                SPDLOG_DEBUG("处理词语：{}", word);
                {
                    size_t iSize = word.size();
                    stream.write(reinterpret_cast<char*>(&iSize), sizeof(size_t));
                    stream.write(word.data(), word.size());
                    size_t xSize = x.size();
                    stream.write(reinterpret_cast<char*>(&xSize), sizeof(size_t));
                    stream.write(reinterpret_cast<char*>(x.data()), xSize * sizeof(float));
                }
            }
        });
    }
    return true;
}
