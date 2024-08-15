#include "lifuren/DocumentReader.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

[[maybe_unused]] static void testPDFReader() {
    lifuren::PDFReader reader{ "D://tmp/word.pdf" };
    SPDLOG_DEBUG("all = {}", reader.readAll());
    reader.reset();
    while(reader.hasMore()) {
        SPDLOG_DEBUG("more = {} = {}", reader.percent(), reader.readMore());
    }
}

[[maybe_unused]] static void testTextReader() {
    lifuren::TextReader reader{ "D://tmp/lifuren.txt" };
    SPDLOG_DEBUG("all = {}", reader.readAll());
    reader.reset();
    while(reader.hasMore()) {
        SPDLOG_DEBUG("more = {} = {}", reader.percent(), reader.readMore());
    }
}

[[maybe_unused]] static void testWordReader() {
    lifuren::WordReader reader{ "D://tmp/word.docx" };
    SPDLOG_DEBUG("all = {}", reader.readAll());
    reader.reset();
    while(reader.hasMore()) {
        SPDLOG_DEBUG("more = {} = {}", reader.percent(), reader.readMore());
    }
}

[[maybe_unused]] static void testMarkdownReader() {
    lifuren::MarkdownReader reader{ "D://tmp/lifuren.txt" };
    SPDLOG_DEBUG("all = {}", reader.readAll());
    reader.reset();
    while(reader.hasMore()) {
        SPDLOG_DEBUG("more = {} = {}", reader.percent(), reader.readMore());
    }
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    // testPDFReader();
    // testTextReader();
    // testWordReader();
    testMarkdownReader();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}