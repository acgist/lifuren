/**
 * 词嵌入终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_NLP_EMBEDDING_CLIENT_HPP
#define LFR_HEADER_NLP_EMBEDDING_CLIENT_HPP

#include <vector>
#include <string>

#include "lifuren/Client.hpp"

namespace lifuren {

/**
 * 词嵌入终端
 */
class EmbeddingClient : public Client {

public:
    EmbeddingClient();
    virtual ~EmbeddingClient();

public:
    /**
     * @param prompt 提示内容
     * 
     * @return 嵌入向量
     */
    virtual std::vector<float> getVector(const std::string& prompt) const = 0;
    /**
     * @param prompts 提示内容
     * 
     * @return 嵌入向量
     */
    virtual std::vector<float> getVector(const std::vector<std::string>& prompts) const;
    /**
     * @return 嵌入向量维度
     */
    virtual size_t getDims() const = 0;

public:
    /**
     * @param embedding 词嵌入类型
     * 
     * @return 词嵌入终端
     */
    static std::unique_ptr<lifuren::EmbeddingClient> getClient(const std::string& embedding);

};

/**
 * Ollama词嵌入终端
 * 
 * https://github.com/ollama/ollama
 */
class OllamaEmbeddingClient : public EmbeddingClient {

private:
    // REST终端
    std::unique_ptr<lifuren::RestClient> restClient{ nullptr };

public:
    OllamaEmbeddingClient();
    virtual ~OllamaEmbeddingClient();

public:
    std::vector<float> getVector(const std::string& prompt) const override;
    size_t getDims() const override;

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
    std::vector<float> getVector(const std::string& prompt) const override;
    size_t getDims() const override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_EMBEDDING_CLIENT_HPP
