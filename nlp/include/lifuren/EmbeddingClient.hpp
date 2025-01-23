/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 词嵌入终端
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_NLP_EMBEDDING_CLIENT_HPP
#define LFR_HEADER_NLP_EMBEDDING_CLIENT_HPP

#include <memory>
#include <string>
#include <vector>

namespace lifuren {

class RestClient;

/**
 * 词嵌入终端
 */
class EmbeddingClient {

protected:
    // 目录
    std::string path;

public:
    EmbeddingClient(const std::string& path);
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
    static std::unique_ptr<lifuren::EmbeddingClient> getClient(const std::string& path, const std::string& embedding);

};

/**
 * Ollama词嵌入终端
 * 
 * https://github.com/ollama/ollama
 * https://github.com/ollama/ollama/blob/main/docs/api.md
 */
class OllamaEmbeddingClient : public EmbeddingClient {

private:
    // REST终端
    std::unique_ptr<lifuren::RestClient> restClient{ nullptr };

public:
    OllamaEmbeddingClient(const std::string& path);
    virtual ~OllamaEmbeddingClient();

public:
    std::vector<float> getVector(const std::string& prompt) const override;
    size_t getDims() const override;

};

/**
 * Pepper（辣椒）词嵌入终端
 * 
 * 自定义词嵌入终端，主要是将第三方词嵌入转为本地文件存储加快词嵌入速度。
 * 格式如下：
 * length(word) word length(vector) vector
 */
class PepperEmbeddingClient : public EmbeddingClient {

public:
    PepperEmbeddingClient(const std::string& path);
    virtual ~PepperEmbeddingClient();

public:
    std::vector<float> getVector(const std::string& prompt) const override;
    size_t getDims() const override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_EMBEDDING_CLIENT_HPP
