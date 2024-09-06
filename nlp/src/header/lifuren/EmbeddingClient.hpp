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
    static std::unique_ptr<lifuren::EmbeddingClient> getClient(const std::string& embedding);

public:
    virtual std::vector<std::vector<float>> getVector(const std::vector<std::string>& segment);
    virtual std::vector<float> getSegmentVector(const std::vector<std::string>& segment);
    virtual std::vector<float> getVector(const std::string& word) = 0;

public:
    EmbeddingClient();
    virtual ~EmbeddingClient();

};

class OllamaEmbeddingClient : public EmbeddingClient {

public:
    OllamaEmbeddingClient();
    virtual ~OllamaEmbeddingClient();

public:
    std::vector<float> getVector(const std::string& word) override;

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

};

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_EMBEDDING_CLIENT_HPP
