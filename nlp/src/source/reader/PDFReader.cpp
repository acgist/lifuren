#include "lifuren/DocumentReader.hpp"

#include <vector>
#include <algorithm>

#include "spdlog/spdlog.h"

lifuren::PDFReader::PDFReader(const std::string& path) : DocumentReader(path) {
    try {
        // TODO: 新的API修改深度
        // PoDoFo::PdfCommon::SetMaxRecursionDepth
        this->document = std::make_unique<PoDoFo::PdfMemDocument>();
        this->document->Load(path);
        const auto& pages = this->document->GetPages();
        this->count = pages.GetCount();
    } catch(const std::exception& e) {
        SPDLOG_ERROR("读取PDF异常：{} - {}", path, e.what());
    }
}

lifuren::PDFReader::~PDFReader() {
}

std::string lifuren::PDFReader::readAll() {
    std::string content;
    const auto& pages = this->document->GetPages();
    for(int index = 0; index < this->count; ++index) {
        const auto& page = pages.GetPageAt(index);
        std::vector<PoDoFo::PdfTextEntry> vector;
        page.ExtractTextTo(vector);
        std::for_each(vector.begin(), vector.end(), [&content](auto& entry) {
            content += entry.Text + '\n';
        });
        content += '\n';
    }
    return content;
}

bool lifuren::PDFReader::hasMore() {
    return this->index < this->count;
}

std::string lifuren::PDFReader::readMore() {
    std::string content;
    const auto& pages = this->document->GetPages();
    const auto& page  = pages.GetPageAt(this->index);
    std::vector<PoDoFo::PdfTextEntry> vector;
    page.ExtractTextTo(vector);
    std::for_each(vector.begin(), vector.end(), [&content](auto& entry) {
        content += entry.Text + '\n';
    });
    content += '\n';
    ++this->index;
    return content;
}

float lifuren::PDFReader::percent() {
    if(this->count <= 0) {
        return 1.0F;
    }
    return static_cast<float>(this->index) / this->count;
}

bool lifuren::PDFReader::reset() {
    this->index = 0;
    return true;
}
