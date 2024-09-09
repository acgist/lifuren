/**
 * 词嵌入终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_NLP_EMBEDDING_CLIENT_HPP
#define LFR_HEADER_NLP_EMBEDDING_CLIENT_HPP

#include <map>
#include <vector>
#include <string>

#include "lifuren/Client.hpp"

namespace lifuren {

/**
 * 词嵌入终端
 */
class EmbeddingClient : public Client {

public:
// 分词类型
enum class SegmentType {
    // 字符
    CHAR,
    // 分词
    WORD,
    // 整句
    SEGMENT,
};

public:
    static std::unique_ptr<lifuren::EmbeddingClient> getClient(const std::string& embedding);

private:
    lifuren::EmbeddingClient::SegmentType type;

public:
    virtual std::vector<float> getSegmentVector(const std::string& segment);
    virtual std::vector<float> getSegmentVector(const std::vector<std::string>& segment);
    virtual std::vector<float> getVector(const std::string& word) = 0;
    virtual std::map<std::string, std::vector<float>> getVector(const std::vector<std::string>& segment);
    virtual size_t getDims() = 0;
    virtual bool release();

public:
    EmbeddingClient(lifuren::EmbeddingClient::SegmentType type);
    virtual ~EmbeddingClient();

};

/**
 * https://github.com/ollama/ollama
 */
class OllamaEmbeddingClient : public EmbeddingClient {

private:
    std::unique_ptr<lifuren::RestClient> restClient{ nullptr };

public:
    OllamaEmbeddingClient();
    virtual ~OllamaEmbeddingClient();

public:
    std::vector<float> getVector(const std::string& word) override;
    size_t getDims() override;

};

/**
 * Chinese-Word-Vectors词嵌入终端
 * 
 * 项目地址：https://github.com/Embedding/Chinese-Word-Vectors
 */
class ChineseWordVectorsEmbeddingClient : public EmbeddingClient {

public:
    ChineseWordVectorsEmbeddingClient();
    virtual ~ChineseWordVectorsEmbeddingClient();

public:
    std::vector<float> getVector(const std::string& word) override;
    size_t getDims() override;
    bool release() override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_EMBEDDING_CLIENT_HPP
