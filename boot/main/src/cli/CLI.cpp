#include "lifuren/CLI.hpp"

#include <mutex>
#include <string>
#include <vector>
#include <iostream>
#include <condition_variable>

#include "spdlog/spdlog.h"

#include "lifuren/RAG.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/Message.hpp"
#include "lifuren/audio/Audio.hpp"
#include "lifuren/EmbeddingClient.hpp"
#include "lifuren/video/ActClient.hpp"
#include "lifuren/image/PaintClient.hpp"
#include "lifuren/audio/ComposeClient.hpp"
#include "lifuren/poetry/PoetizeClient.hpp"

static void act(      const std::vector<std::string>&); // 视频生成
static void paint(    const std::vector<std::string>&); // 图片生成
static void compose(  const std::vector<std::string>&); // 音频生成
static void poetize(  const std::vector<std::string>&); // 诗词生成
static void pcm(      const std::vector<std::string>&); // 转为PCM
static void pepper(   const std::vector<std::string>&); // 辣椒嵌入
static void embedding(const std::vector<std::string>&); // 诗词嵌入
static void messageCallback(bool, const char*); // 消息回调
static void help(); // 帮助

// TODO: 注册回调

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
    } else if(std::strcmp(command, "pcm") == 0) {
        pcm(args);
    } else if(std::strcmp(command, "pepper") == 0) {
        pepper(args);
    } else if(std::strcmp(command, "embedding") == 0) {
        embedding(args);
    } else {
        SPDLOG_WARN("不支持的命令：{}", command);
    }
    return true;
}

static void act(const std::vector<std::string>& args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::getActClient(lifuren::config::CONFIG.video.client);
    // TODO: 实现
 }

static void paint(const std::vector<std::string>& args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::getPaintClient(lifuren::config::CONFIG.image.client);
    // TODO: 实现
}

static void compose(const std::vector<std::string>& args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::getComposeClient(lifuren::config::CONFIG.audio.client);
    // TODO: 实现
}

static void poetize(const std::vector<std::string>& args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::getPoetizeClient(lifuren::config::CONFIG.poetry.client);
    // TODO: 实现
}

static void pcm(const std::vector<std::string>& args) {
    lifuren::message::registerMessageCallback(lifuren::message::Type::AUDIO_AUDIO_FILE_TO_PCM_FILE, messageCallback);
    if(args.empty()) {
        lifuren::message::sendMessage("缺少参数");
        return;
    }
    auto preprocessing = std::bind(&lifuren::audio::preprocessing, std::placeholders::_1);
    if(lifuren::dataset::allDatasetPreprocessing(args[0], preprocessing)) {
        lifuren::message::sendMessage("PCM转换成功");
    } else {
        lifuren::message::sendMessage("PCM转换失败");
    }
    lifuren::message::unregisterMessageCallback(lifuren::message::Type::AUDIO_AUDIO_FILE_TO_PCM_FILE);
}

static void pepper(const std::vector<std::string>& args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = std::make_unique<lifuren::PepperEmbeddingClient>();
    auto embedding = std::bind(&lifuren::PepperEmbeddingClient::embedding, client.get(), std::placeholders::_1);
    // auto embedding = std::bind(&lifuren::PepperEmbeddingClient::embedding, std::ref(*client), std::placeholders::_1);
    if(lifuren::dataset::allDatasetPreprocessing(args[0], embedding)) {
        SPDLOG_INFO("辣椒嵌入成功");
    } else {
        SPDLOG_WARN("辣椒嵌入失败");
    }
}

static void embedding(const std::vector<std::string>& args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    if(lifuren::RAGClient::rag(args[0], args[1], args[2])) {
        SPDLOG_INFO("嵌入成功");
    } else {
        SPDLOG_WARN("嵌入失败");
    }
}

static void messageCallback(bool finish, const char* message) {
    std::cout << message << std::endl;
    if(finish) {
        std::cout << "任务完成" << std::endl;
    }
}

static void help() {
    std::cout << R"(
./lifuren[.exe] 命令 [参数...]
./lifuren[.exe] act       [train|pred] [model|dataset] file
./lifuren[.exe] paint     [train|pred] [model|dataset] file
./lifuren[.exe] compose   [train|pred] [model|dataset] file
./lifuren[.exe] poetize   [train|pred] [model|dataset] prompt
./lifuren[.exe] pcm       path
./lifuren[.exe] pepper    path
./lifuren[.exe] embedding path
./lifuren[.exe] [?|help]
    )" << std::endl;
}
