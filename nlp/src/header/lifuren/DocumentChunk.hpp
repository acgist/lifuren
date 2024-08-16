/**
 * 文档分段
 * 
 * TODO:
 * 1. 添加智能分段：通过词频和稀疏度
 */
#ifndef LIFUREN_HEADER_NLP_DOCUMENT_CHUNK_HPP
#define LIFUREN_HEADER_NLP_DOCUMENT_CHUNK_HPP

#include <list>

#include "lifuren/DocumentReader.hpp"

namespace lifuren {

/**
 * 分段模式
 */
enum class ChunkType {

    // 按行分段
    LINE,
    // 标题分段
    TITLE,

};

/**
 * 分段策略
 */
class ChunkStrategy {

protected:
    // 分段模式
    const ChunkType type;
    // 文档内容
    std::string document;

public:
    /**
     * @param type 分段模式
     */
    ChunkStrategy(ChunkType type);
    virtual ~ChunkStrategy();

public:
    /**
     * @return 分段模式
     */
    ChunkType chunkType();
    /**
     * @param content 文档内容
     * 
     * @return 分段内容
     */
    virtual std::list<std::string> chunk(const std::string& content, bool last = false) = 0;
    /**
     * @param chunkType 分段类型
     * 
     * @return 分段策略
     */
    static std::unique_ptr<lifuren::ChunkStrategy> getChunkStrategy(const std::string& chunkType);

};

/**
 * 按行分段
 */
class LineChunkStrategy : public ChunkStrategy {

public:
    LineChunkStrategy();
    virtual ~LineChunkStrategy();

public:
    virtual std::list<std::string> chunk(const std::string& content, bool last = false) override;

};

/**
 * 分段服务
 */
class ChunkService {

public:
    // 分段策略
    std::unique_ptr<lifuren::ChunkStrategy> chunkStrategy{ nullptr };

public:
    ChunkService(const std::string& chunkType);
    ~ChunkService();

public:
    /**
     * 执行分段
     * 
     * @param path 文件路径
     * 
     * @return 分段内容
     */
    std::list<std::string> chunk(const std::string& path);

};

} // END OF lifuren

#endif // END OF LIFUREN_HEADER_NLP_DOCUMENT_CHUNK_HPP
