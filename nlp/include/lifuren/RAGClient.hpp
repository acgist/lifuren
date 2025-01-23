/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * RAG（检索增强生成）
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LIFUREN_HEADER_NLP_RAG_HPP
#define LIFUREN_HEADER_NLP_RAG_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace faiss {
    
    struct Index;

}

namespace lifuren {

class RestClient;
class EmbeddingClient;

/**
 * RAG任务
 */
struct RAGTask {
    
    // RAG方式：faiss|elasticsearch
    std::string rag;
    // 文档路径
    std::string path;
    // 词嵌入方式：ollama|pepper
    std::string embedding;

};

/**
 * RAG搜索终端
 */
class RAGSearchClient {

public:
    /**
     * @return 文档内容
     */
    virtual std::vector<std::string> search(
        const std::string& prompt,  // 搜索内容
        const uint8_t      size = 4 // 结果数量
    ) const = 0;
    /**
     * @return 文档内容
     */
    virtual std::vector<std::string> search(
        const std::vector<float>& prompt,  // 搜索向量
        const uint8_t             size = 4 // 结果数量
    ) const = 0;

};

/**
 * RAG终端
 * 
 * 提供词嵌入和词向量搜索
 * 
 * 注意：
 *  1. 只能保证单次索引提示不会重复
 *  2. 多次索引提示建议先删除再重建
 */
class RAGClient : public RAGSearchClient {

public:
    // RAG文档ID
    size_t id = 10001000;

protected:
    // 文档路径：pepper.model/indexDB.model/mapping.model
    std::string path;
    // 数据集路径：embedding.model
    std::string dataset_path;
    // 词嵌入终端
    std::unique_ptr<lifuren::EmbeddingClient> embeddingClient{ nullptr };

public:
    /**
     * @param path      文档路径
     * @param embedding 词嵌入方式
     */
    RAGClient(const std::string& path, const std::string& embedding);
    virtual ~RAGClient();

protected:
    /**
     * 是否已经索引提示内容
     * 
     * @param prompt 提示内容
     * 
     * @return 是否已经索引
     */
    bool donePromptEmplace(const std::string& prompt);

public:
    /**
     * @return 嵌入向量维度
     */
    virtual size_t getDims() const;
    /**
     * 判断是否重复索引，但是不要缓存，缓存在EmbeddingClient里面实现。
     * 
     * @param prompt 提示内容
     * 
     * @return 提示向量
     */
    virtual std::vector<float> index(const std::string& prompt) = 0;
    
    virtual std::vector<std::string> search(const std::string& prompt,        const uint8_t size = 4) const override;
    virtual std::vector<std::string> search(const std::vector<float>& prompt, const uint8_t size = 4) const override = 0;

public:
    /**
     * @param rag       RAG类型
     * @param path      文档路径
     * @param embedding 词嵌入方式
     * 
     * @return RAG终端
     */
    static std::unique_ptr<lifuren::RAGClient> getClient(const std::string& rag, const std::string& path, const std::string& embedding);
    /**
     * @param task RAG任务
     * 
     * @return RAG终端
     */
    inline static std::unique_ptr<lifuren::RAGClient> getClient(const RAGTask& task) {
        return RAGClient::getClient(task.rag, task.path, task.embedding);
    }

};

/**
 * FaissRAG终端
 * 
 * https://github.com/facebookresearch/faiss
 * https://github.com/facebookresearch/faiss/tree/main/tutorial/cpp
 */
class FaissRAGClient : public RAGClient {

protected:
    // 向量数据库
    std::shared_ptr<faiss::Index> indexDB{ nullptr };
    // ID = 单词
    std::shared_ptr<std::map<size_t, std::string>> mapping{ nullptr };

public:
    /**
     * @param path      文档路径
     * @param embedding 词嵌入方式
     */
    FaissRAGClient(const std::string& path, const std::string& embedding);
    virtual ~FaissRAGClient();

public:
    using RAGClient::search;
    std::vector<float> index(const std::string& prompt) override;
    std::vector<std::string> search(const std::vector<float>& prompt, const uint8_t size = 4) const override;

};

/**
 * ElasticSearchRAG终端
 * 
 * https://github.com/elastic/elasticsearch
 * https://www.elastic.co/guide/en/elasticsearch/reference/current/rest-apis.html
 */
class ElasticSearchRAGClient : public RAGClient {

protected:
    // REST终端
    std::shared_ptr<lifuren::RestClient> restClient{ nullptr };

public:
    /**
     * @param path      文档路径
     * @param embedding 词嵌入方式
     */
    ElasticSearchRAGClient(const std::string& path, const std::string& embedding);
    virtual ~ElasticSearchRAGClient();

public:
    using RAGClient::search;
    std::vector<float> index(const std::string& prompt) override;
    std::vector<std::string> search(const std::vector<float>& prompt, const uint8_t size = 4) const override;

};

} // END OF lifuren

#endif // END OF LIFUREN_HEADER_NLP_RAG_HPP
