/**
 * RAG
 */
#ifndef LIFUREN_HEADER_NLP_RAG_HPP
#define LIFUREN_HEADER_NLP_RAG_HPP

#include <map>
#include <string>
#include <thread>
#include <vector>

#include "lifuren/Client.hpp"
#include "lifuren/DocumentChunk.hpp"

namespace lifuren {

/**
 * RAG任务
 */
struct RAGTask {
    
    // 文档路径
    std::string path;
    // RAG方式
    std::string rag;
    // 分段方式
    std::string chunk;
    // 词嵌入方式
    std::string embedding;

};

/**
 * 词向量服务
 */
class EmbeddingService {

private:
    // 词嵌入终端
    std::unique_ptr<EmbeddingClient> embeddingClient{ nullptr };

public:
    EmbeddingService(const std::string& embedding);
    ~EmbeddingService();

};

/**
 * RAG终端
 */
class RAGClient {

protected:
    std::unique_ptr<lifuren::EmbeddingService> embeddingService{ nullptr };

public:
    RAGClient(const std::string& embedding);
    ~RAGClient();

public:
    /**
     * 索引建立
     */
    virtual std::vector<double> index(const std::string& content) = 0;
    /**
     * 索引搜索
     */
    virtual std::string search(const std::vector<double>& vector) = 0;

};

class FaissRAGClient : public RAGClient {
};

class MilvusRAGClient : public RAGClient {
};

class ChromaRAGClient : public RAGClient {
};

class TypesenseRAGClient : public RAGClient {
};

class ElasticSearchRAGClient : public RAGClient {

public:
    ElasticSearchRAGClient(const std::string& embedding);
    ~ElasticSearchRAGClient();

public:
    virtual std::vector<double> index(const std::string& content) override;
    virtual std::string search(const std::vector<double>& vector) override;

};

/**
 * RAG任务执行器
 */
class RAGTaskRunner {

public:
    // 是否完成
    bool finish = false;

protected:
    // 文件总数
    uint32_t fileCount = 0;
    // 失败处理文件总数
    uint32_t failFileCount = 0;
    // 成功处理文件总数
    uint32_t successFileCount = 0;
    // RAG任务
    RAGTask task;
    // 执行线程
    std::unique_ptr<std::thread> thread{ nullptr };
    // RAG终端
    std::unique_ptr<lifuren::RAGClient> ragClient{ nullptr };
    // 分段服务
    std::unique_ptr<lifuren::ChunkService> chunkService{ nullptr };

public:
    RAGTaskRunner(RAGTask task);
    virtual ~RAGTaskRunner();

public:
    // 执行任务
    bool execute();
    // 任务进度
    float percent();

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
    ~RAGService();

private:
    // RAG任务列表
    std::map<std::string, RAGTaskRunner> tasks;

public:
    /**
     * 添加任务
     * 
     * @param task RAG任务
     * 
     * @return 是否成功
     */
    bool buildRAGTask(RAGTask task);
    /**
     * 结束任务
     * 
     * @param path 任务路径
     * 
     * @return 是否成功
     */
    bool stopRAGTask(const std::string& path);
    /**
     * @return 当前任务总量（执行中和待执行的总量）
     */
    int taskCount();
    /**
     * @param path 任务路径
     * 
     * @return 任务进度
     */
    float taskPercent(const std::string& path);

};

} // END OF lifuren

#endif // END OF LIFUREN_HEADER_NLP_RAG_HPP
