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
    class Index;
}

namespace lifuren {

/**
 * RAG任务
 */
struct RAGTask {
    
    // RAG方式
    std::string type;
    // 文档路径
    std::string path;
    // 词嵌入方式
    std::string embedding;

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
    virtual void loadIndex();
    // 保存索引
    virtual void saveIndex();
    // 清空索引
    virtual void truncateIndex();
    /**
     * 添加已经处理文件
     * 
     * @param file 处理文件路径
     */
    bool doneFileEmplace(const std::string& file);
    /**
     * @param content 文档内容
     * 
     * @return 索引向量
     */
    virtual std::vector<float> index(const std::string& content) = 0;
    
    virtual std::vector<std::string> search(const std::string& prompt, const int size = 4) override;
    virtual std::vector<std::string> search(const std::vector<float>& prompt, const int size = 4) = 0;

public:
    /**
     * @param type      RAG终端类型
     * @param path      文档路径
     * @param embedding 词嵌入方式
     * 
     * @return RAG终端
     */
    static std::unique_ptr<lifuren::RAGClient> getRAGClient(const std::string& type, const std::string& path, const std::string& embedding);

};

/**
 * FaissRAG终端
 * 
 * https://github.com/facebookresearch/faiss
 */
class FaissRAGClient : public RAGClient {

protected:
    std::shared_ptr<faiss::Index> indexBasicDB { nullptr };
    std::shared_ptr<faiss::Index> indexIdMapDB { nullptr };
    std::shared_ptr<std::map<size_t, std::string>> idMapping{ nullptr };

public:
    /**
     * @param path      文档路径
     * @param embedding 词嵌入方式
     */
    FaissRAGClient(const std::string& path, const std::string& embedding);
    ~FaissRAGClient();

public:
    using RAGClient::search;
    std::vector<float> index(const std::string& content) override;
    std::vector<std::string> search(const std::vector<float>& prompt, const int size = 4) override;
    void truncateIndex() override;

};

/**
 * ElasticSearchRAG终端
 * 
 * https://github.com/elastic/elasticsearch
 */
class ElasticSearchRAGClient : public RAGClient {

protected:
    // 是否存在索引
    bool exists = false;
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
    std::vector<float> index(const std::string& content) override;
    std::vector<std::string> search(const std::vector<float>& prompt, const int size = 4) override;
    void truncateIndex() override;

};

/**
 * RAG任务执行器
 */
class RAGTaskRunner {

public:
    // 索引标识
    size_t id = 0LL;
    // 是否停止
    bool stop = false;
    // 是否完成
    bool finish = false;

private:
    // 加锁
    std::mutex mutex;
    // 文件总数
    uint32_t fileCount = 0;
    // 处理文件总数
    uint32_t doneFileCount = 0;
    // RAG任务
    RAGTask task;
    // 执行线程
    std::unique_ptr<std::thread> thread{ nullptr };
    // RAG终端
    std::unique_ptr<lifuren::RAGClient> ragClient{ nullptr };
    // 进度回调
    std::function<void(float, bool)> percentCallback{ nullptr };

public:
    RAGTaskRunner(RAGTask task);
    virtual ~RAGTaskRunner();

private:
    // 执行任务
    bool execute();

public:
    /**
     * 开始任务
     * 
     * @return 是否成功
     */
    bool startExecute();
    /**
     * @return 任务进度
     */
    float percent();
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
    /**
     * RAG任务执行器列表
     * 任务地址 = RAG任务执行器
     */
    std::map<std::string, std::shared_ptr<RAGTaskRunner>> tasks{};

public:
    /**
     * @param path 任务路径
     * 
     * @return RAG任务执行器
     */
    std::shared_ptr<RAGTaskRunner> getRAGTask(const std::string& path);
    /**
     * @param path 任务路径
     * 
     * @return RAG任务执行器
     */
    std::shared_ptr<RAGTaskRunner> runRAGTask(const std::string& path);
    /**
     * 结束任务
     * 
     * @param path 任务路径
     * 
     * @return 是否成功
     */
    bool stopRAGTask(const std::string& path);
    /**
     * 移除任务
     * 
     * @param path 任务路径
     * 
     * @return 是否成功
     */
    bool removeRAGTask(const std::string& path);
    /**
     * @return 当前任务总量（执行中和待执行的总量）
     */
    size_t taskCount();

};

} // END OF lifuren

#endif // END OF LIFUREN_HEADER_NLP_RAG_HPP
