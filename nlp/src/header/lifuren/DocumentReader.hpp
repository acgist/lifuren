/**
 * 文档解析：TXT/PDF/Markdown/Word2007+
 * 
 * 注意：
 * 1. PDF没有实现OCR功能
 * 2. 
 * 
 * 其他阅读器推荐：
 * 1. https://github.com/opendatalab/MinerU
 * 
 * TODO:
 * 1. 优化阅读性能
 * 2. 优化阅读效果
 * 3. PDF/Word公式支持
 */
#ifndef LIFUREN_HEADER_NLP_DOCUMENT_READER_HPP
#define LIFUREN_HEADER_NLP_DOCUMENT_READER_HPP

#include <memory>
#include <string>
#include <fstream>

#include "minidocx.hpp"

#include "podofo/podofo.h"

namespace lifuren {

/**
 * 文档阅读器
 */
class DocumentReader {

public:
    // 文档路径
    const std::string path;

public:
    /**
     * @param path 文档路径
     */
    DocumentReader(const std::string& path);
    virtual ~DocumentReader();

public:
    /**
     * 读取全部文档内容
     * 
     * @return 文档内容
     */
    virtual std::string readAll() = 0;
    /**
     * @return 是否还有更多文档内容
     */
    virtual bool hasMore() = 0;
    /**
     * 读取更多文档内容
     * 
     * 必须配合`hasMore()`方法使用。
     * 不会读取全部内容，不同文档类型读取内容方式不同。
     * 
     * @return 文档内容
     */
    virtual std::string readMore() = 0;
    /**
     * @return 读取进度
     */
    virtual float percent() = 0;
    /**
     * 重置状态
     * 
     * @return 是否成功
     */
    virtual bool reset() = 0;
    /**
     * @param path 文件路径
     * 
     * @return 文档阅读器
     */
    static std::unique_ptr<lifuren::DocumentReader> getReader(const std::string& path);

};

/**
 * PDF读取
 * 
 * 项目地址：https://github.com/podofo/podofo
 */
class PDFReader : public DocumentReader {

public:
    // 当前页码
    int index = 0;
    // 总的页码
    int count = 0;
    // PDF文档
    std::unique_ptr<PoDoFo::PdfMemDocument> document{ nullptr };

public:
    /**
     * @param path 文档路径
     */
    PDFReader(const std::string& path);
    virtual ~PDFReader();

public:
    virtual std::string readAll()  override;
    virtual bool        hasMore()  override;
    virtual std::string readMore() override;
    virtual float       percent()  override;
    virtual bool        reset()    override;

};

/**
 * TXT读取
 */
class TextReader : public DocumentReader {

public:
    // 文档大小
    size_t fileSize = 0L;
    // 文档流
    std::ifstream input;

public:
    /**
     * @param path 文档路径
     */
    TextReader(const std::string& path);
    virtual ~TextReader();

public:
    virtual std::string readAll()  override;
    virtual bool        hasMore()  override;
    virtual std::string readMore() override;
    virtual float       percent()  override;
    virtual bool        reset()    override;

};

/**
 * Word读取
 * 
 * 项目地址：https://github.com/totravel/minidocx
 */
class WordReader : public DocumentReader {

public:
    // Word文档
    std::unique_ptr<docx::Document> document{ nullptr };
    // 当前段落
    docx::Paragraph paragraph;

public:
    /**
     * @param path 文档路径
     */
    WordReader(const std::string& path);
    virtual ~WordReader();

public:
    virtual std::string readAll()  override;
    virtual bool        hasMore()  override;
    virtual std::string readMore() override;
    virtual float       percent()  override;
    virtual bool        reset()    override;

};

/**
 * Markdown读取
 */
class MarkdownReader : public DocumentReader {

public:
    // 文档大小
    size_t fileSize = 0L;
    // 文档流
    std::ifstream input;

public:
    /**
     * @param path 文档路径
     */
    MarkdownReader(const std::string& path);
    virtual ~MarkdownReader();

public:
    virtual std::string readAll()  override;
    virtual bool        hasMore()  override;
    virtual std::string readMore() override;
    virtual float       percent()  override;
    virtual bool        reset()    override;

};

} // END OF lifuren

#endif // END OF LIFUREN_HEADER_NLP_DOCUMENT_READER_HPP
