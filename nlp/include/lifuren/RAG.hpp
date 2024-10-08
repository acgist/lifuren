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
    // 词嵌入方式：ollama|chinese-word-vectors
    std::string embedding;
    // 文档路径
    std::string path;

};

/**
 * RAG终端
 */
class RAGClient : public RAGSearchClient {

public:
    // 唯一标识
    size_t id = 0LL;

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

public:
    // 加载索引
    virtual bool loadIndex();
    // 保存索引
    virtual bool saveIndex();
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
     * @param prompt 提示内容
     * 
     * @return 提示向量
     */
    virtual std::vector<float> index(const std::string& prompt) = 0;
    
    virtual std::vector<std::string> search(const std::string& prompt,        const int size = 4) const override;
    virtual std::vector<std::string> search(const std::vector<float>& prompt, const int size = 4) const override = 0;

public:
    /**
     * @param rag       RAG类型
     * @param embedding 词嵌入方式
     * @param path      文档路径
     * 
     * @return RAG终端
     */
    static std::unique_ptr<lifuren::RAGClient> getClient(const std::string& rag, const std::string& embedding, const std::string& path);

};

/**
 * FaissRAG终端
 * 
 * https://github.com/facebookresearch/faiss
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
    ~FaissRAGClient();

public:
    using RAGClient::search;
    std::vector<float> index(const std::string& prompt) override;
    std::vector<std::string> search(const std::vector<float>& prompt, const int size = 4) const override;
    bool loadIndex() override;
    bool saveIndex() override;
    bool truncateIndex() override;

};

/**
 * ElasticSearchRAG终端
 * 
 * https://github.com/elastic/elasticsearch
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
    ~ElasticSearchRAGClient();

public:
    using RAGClient::search;
    std::vector<float> index(const std::string& prompt) override;
    std::vector<std::string> search(const std::vector<float>& prompt, const int size = 4) const override;
    bool loadIndex() override;
    bool truncateIndex() override;

};

/**
 * RAG任务执行器
 * 
 * 1. 诗词分词
 * 2. 生成词嵌入文件
 * 3. 生成向量搜索文件
 */
class RAGTaskRunner {

public:
    size_t id   = 0LL;   // 索引标识
    bool stop   = false; // 是否停止
    bool finish = false; // 是否完成

private:
    // RAG任务
    RAGTask task;
    // 互斥锁
    std::mutex mutex;
    // 文件总数
    uint32_t fileCount = 0;
    // 处理文件总数
    uint32_t doneFileCount = 0;
    // 执行线程
    std::unique_ptr<std::thread> thread{ nullptr };
    // RAG终端
    std::unique_ptr<lifuren::RAGClient> ragClient{ nullptr };
    // 进度回调
    std::function<void(float, bool)> percentCallback{ nullptr };

public:
    /**
     * @param task RAG任务
     */
    RAGTaskRunner(RAGTask task);
    virtual ~RAGTaskRunner();

private:
    /**
     * 执行任务
     * 
     * @return 是否成功
     */
    bool execute();

public:
    /**
     * @return 是否成功
     */
    bool startExecute();
    /**
     * @return 任务进度
     */
    float percent() const;
    /**
     * 注册进度回调
     * 
     * @param percentCallback 进度回调
     */
    void registerCallback(std::function<void(float, bool)> percentCallback);
    /**
     * 取消进度回调注册
     */
    void unregisterCallback();

};

/**
 * RAG服务
 * 
 * 文档解析、文档分段、文档搜索
 */
class RAGService {

public:
    /**
     * @return 单例
     */
    static RAGService& getInstance();

private:
    RAGService();

public:
    RAGService(RAGService& ) = delete;
    RAGService(RAGService&&) = delete;
    RAGService operator=(RAGService& ) = delete;
    RAGService operator=(RAGService&&) = delete;
    ~RAGService();

private:
    // 任务地址 = RAG任务执行器
    std::map<std::string, std::shared_ptr<RAGTaskRunner>> taskMap{};

public:
    /**
     * @param path 任务路径
     * 
     * @return RAG任务执行器
     */
    std::shared_ptr<RAGTaskRunner> getRAGTask(const std::string& path) const;
    /**
     * @param path 任务路径
     * 
     * @return RAG任务执行器
     */
    std::shared_ptr<RAGTaskRunner> runRAGTask(const std::string& path);
    /**
     * @param path 任务路径
     * 
     * @return 是否成功
     */
    bool stopRAGTask(const std::string& path) const;
    /**
     * @param path 任务路径
     * 
     * @return 是否成功
     */
    bool removeRAGTask(const std::string& path);
    /**
     * @return 当前任务总量（执行中和待执行的总量）
     */
    size_t taskCount() const;

};

} // END OF lifuren

#endif // END OF LIFUREN_HEADER_NLP_RAG_HPP
