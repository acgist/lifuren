#include "lifuren/DocumentReader.hpp"

#include "spdlog/spdlog.h"

lifuren::WordReader::WordReader(const std::string& path) : DocumentReader(path) {
    this->document = std::make_unique<docx::Document>();
    if(this->document->Open(path)) {
        this->paragraph = this->document->FirstParagraph();
    } else {
        SPDLOG_WARN("Word打开失败：{}", path);
    }
}

lifuren::WordReader::~WordReader() {
}

std::string lifuren::WordReader::readAll() {
    std::string content;
    auto paragraph = this->document->FirstParagraph();
    while(paragraph) {
        auto run = paragraph.FirstRun();
        while(run) {
            content += run.GetText();
            run = run.Next();
        }
        content += '\n';
        paragraph = paragraph.Next();
    }
    return content;
}

bool lifuren::WordReader::hasMore() {
    return this->paragraph;
}

std::string lifuren::WordReader::readMore() {
    std::string content;
    auto run = this->paragraph.FirstRun();
    while(run) {
        content += run.GetText();
        run = run.Next();
    }
    content += '\n';
    this->paragraph = paragraph.Next();
    return content;
}

float lifuren::WordReader::percent() {
    // TODO: 实现计算
    return 0.0F;
}

bool lifuren::WordReader::reset() {
    this->paragraph = this->document->FirstParagraph();
    return true;
}
