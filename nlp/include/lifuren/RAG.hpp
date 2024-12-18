/**
 * RAG（检索增强生成）
 */
#ifndef LIFUREN_HEADER_NLP_RAG_HPP
#define LIFUREN_HEADER_NLP_RAG_HPP

#include <map>
#include <set>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <functional>

#include "lifuren/EmbeddingClient.hpp"

namespace faiss {
    struct Index;
}

namespace lifuren {

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
    size_t id = 0;

protected:
    // 文档路径
    std::string path;
    // 处理完成文件列表
    std::set<std::string> doneFile;
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
    // 加载索引
    virtual bool loadIndex();
    // 保存索引
    virtual bool saveIndex() const;
    // 清空索引
    virtual bool truncateIndex();
    /**
     * @return 嵌入向量维度
     */
    virtual size_t getDims() const;
    /**
     * @param file 处理文件路径
     * 
     * @return 是否已经添加
     */
    bool doneFileEmplace(const std::string& file);
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

    static bool rag(const std::string& rag, const std::string& path, const std::string& embedding);

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
    bool loadIndex() override;
    bool saveIndex() const override;
    bool truncateIndex() override;

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
    bool loadIndex() override;
    bool truncateIndex() override;

};

} // END OF lifuren

#endif // END OF LIFUREN_HEADER_NLP_RAG_HPP
