#include "lifuren/CLI.hpp"

#include <mutex>
#include <string>
#include <vector>
#include <iostream>
#include <condition_variable>

#include "spdlog/spdlog.h"

#include "lifuren/RAG.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/video/ActClient.hpp"
#include "lifuren/image/PaintClient.hpp"
#include "lifuren/audio/ComposeClient.hpp"
#include "lifuren/poetry/PoetizeClient.hpp"

static void act(      std::vector<std::string>); // 视频生成
static void paint(    std::vector<std::string>); // 图片生成
static void compose(  std::vector<std::string>); // 音频生成
static void poetize(  std::vector<std::string>); // 诗词生成
static void embedding(std::vector<std::string>); // 诗词嵌入
static void help();      // 帮助

bool lifuren::cli(const int argc, const char* const argv[]) {
    if(argc <= 1) {
        return false;
    }
    for(int i = 0; i < argc; ++i) {
        SPDLOG_DEBUG("命令参数：{}", argv[i]);
    }
    // 命令
    const char* const command = argv[1];
    // 参数
    std::vector<std::string> args;
    for(int i = 2; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    if(
        std::strcmp(command, "?")    == 0 ||
        std::strcmp(command, "help") == 0
    ) {
        help();
    } else if(std::strcmp(command, "act") == 0) {
        act(args);
    } else if(std::strcmp(command, "paint") == 0) {
        paint(args);
    } else if(std::strcmp(command, "compose") == 0) {
        compose(args);
    } else if(std::strcmp(command, "poetize") == 0) {
        poetize(args);
    } else if(std::strcmp(command, "embedding") == 0) {
        embedding(args);
    } else {
        SPDLOG_WARN("不支持的命令：{}", command);
    }
    return true;
}

static void act(std::vector<std::string> args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::ActClient::getClient(lifuren::config::CONFIG.video.client);
    // TODO: 实现
 }

static void paint(std::vector<std::string> args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::PaintClient::getClient(lifuren::config::CONFIG.image.client);
    // TODO: 实现
}

static void compose(std::vector<std::string> args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::ComposeClient::getClient(lifuren::config::CONFIG.audio.client);
    // TODO: 实现
}

static void poetize(std::vector<std::string> args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::PoetizeClient::getClient(lifuren::config::CONFIG.poetry.client);
    // TODO: 实现
}

static void embedding(std::vector<std::string> args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto& service = lifuren::RAGService::getInstance();
    auto  ragTask = service.runRAGTask(args[0]);
    if(!ragTask) {
        return;
    }
    std::mutex mutex;
    std::condition_variable condition;
    ragTask->registerCallback([&mutex, &condition](float, bool finish) {
        if(finish) {
            std::lock_guard<std::mutex> lock(mutex);
            condition.notify_one();
        } else {
            // 进度
        }
    });
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock);
    std::cout << "成功" << std::endl;
}

static void help() {
    std::cout << R"(
./lifuren[.exe] 命令 [参数...]
./lifuren[.exe] act     video
./lifuren[.exe] paint   image
./lifuren[.exe] compose audio
./lifuren[.exe] poetize prompt
./lifuren[.exe] embedding
./lifuren[.exe] [?|help]
    )" << std::endl;
}
