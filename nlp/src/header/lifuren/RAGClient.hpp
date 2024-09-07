/**
 * RAG（检索增强生成）终端
 * 
 * 提供文档索引建立、文档内容搜索
 */
#ifndef LIFUREN_HEADER_NLP_RAG_CLIENT_HPP
#define LIFUREN_HEADER_NLP_RAG_CLIENT_HPP

#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <functional>

#include "lifuren/EmbeddingClient.hpp"

namespace lifuren {

/**
 * RAG任务
 */
struct RAGTask {
    
    // RAG方式
    std::string rag;
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
    size_t id = 0L;
protected:
    // 处理文件列表
    std::set<std::string> doneFile;
    // 文档路径
    std::string path;
    // 词嵌入服务
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
    void loadIndex();
    // 保存索引
    void saveIndex();
    // 清空索引
    void truncateIndex();
    // 添加已经处理文件
    void doneFileEmplace(const std::string& file);
    // 文件是否已经处理
    bool doneFileContains(const std::string& file);
    /**
     * 索引建立
     * 
     * @param content 文档内容
     * 
     * @return 索引内容
     */
    virtual std::vector<float> index(const std::string& content) = 0;
    /**
     * 删除索引
     */
    virtual bool deleteRAG() = 0;
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
 * https://github.com/facebookresearch/faiss
 */
class FaissRAGClient : public RAGClient {

public:

public:
    FaissRAGClient(const std::string& path, const std::string& embedding);
    ~FaissRAGClient();

public:
    std::vector<float> index(const std::string& content) override;
    std::vector<std::string> search(const std::string& prompt) override;
    bool deleteRAG() override;

};

/**
 * ElasticSearchRAG终端
 * 
 * https://github.com/elastic/elasticsearch
 */
class ElasticSearchRAGClient : public RAGClient {

public:
    // 索引是否存在
    bool exists = false;
    // REST终端
    std::shared_ptr<lifuren::RestClient> restClient{ nullptr };

public:
    ElasticSearchRAGClient(const std::string& path, const std::string& embedding);
    ~ElasticSearchRAGClient();

public:
    std::vector<float> index(const std::string& content) override;
    std::vector<std::string> search(const std::string& prompt) override;
    bool deleteRAG() override;

};

/**
 * RAG任务执行器
 */
class RAGTaskRunner {

public:
    // 索引
    size_t id = 0L;
    // 是否停止
    bool stop = false;
    // 是否完成
    bool finish = false;

protected:
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
    // 开始任务
    bool startExecute();
    // 删除任务
    bool deleteRAG();
    // 任务进度
    float percent();
    /**
     * 注册进度回调
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
    // RAG任务列表
    std::map<std::string, std::shared_ptr<RAGTaskRunner>> tasks;

public:
    /**
     * 添加任务
     * 
     * @param path 任务路径
     * 
     * @return RAG任务
     */
    std::shared_ptr<RAGTaskRunner> getRAGTask(const std::string& path);
    /**
     * 添加任务
     * 
     * @param path 任务路径
     * 
     * @return 是否成功
     */
    std::shared_ptr<RAGTaskRunner> buildRAGTask(const std::string& path);
    /**
     * 结束任务
     * 
     * @param path 任务路径
     * 
     * @return 是否成功
     */
    bool stopRAGTask(const std::string& path);
    /**
     * 删除任务
     * 
     * @param path 任务路径
     * 
     * @return 是否成功
     */
    bool deleteRAGTask(const std::string& path);
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

#endif // END OF LIFUREN_HEADER_NLP_RAG_CLIENT_HPP
