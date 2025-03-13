#include "lifuren/CLI.hpp"

#include <string>
#include <vector>
#include <iostream>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Message.hpp"
#include "lifuren/audio/Audio.hpp"
#include "lifuren/image/Image.hpp"

static void generateAudio(const std::vector<std::string>&); // 生成音频
static void generateImage(const std::vector<std::string>&); // 生成视频
static void embedding    (const std::vector<std::string>&); // 数据嵌入
static void help(); // 帮助
static void messageCallback(bool, const char*); // 消息回调

bool lifuren::cli(const int argc, const char* const argv[]) {
    if(argc <= 1) {
        return false;
    }
    lifuren::message::registerMessageCallback(messageCallback);
    for(int i = 0; i < argc; ++i) {
        SPDLOG_DEBUG("命令参数：{} {}", i, argv[i]);
    }
    const char* const command = argv[1];
    std::vector<std::string> args;
    for(int i = 2; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    if(
        std::strcmp(command, "?")    == 0 ||
        std::strcmp(command, "help") == 0
    ) {
        ::help();
    } else if(std::strcmp(command, "audio") == 0) {
        ::generateAudio(args);
    } else if(std::strcmp(command, "image") == 0) {
        ::generateImage(args);
    } else if(std::strcmp(command, "embedding") == 0) {
        ::embedding(args);
    } else {
        SPDLOG_WARN("不支持的命令：{}", command);
        ::help();
    }
    lifuren::message::unregisterMessageCallback();
    return true;
}

// ./lifuren[.exe] audio [bach|shikuang|beethoven] [pred|train] model_file [audio_file|dataset]
static void generateAudio(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    const auto& client_name = args[0];
    auto client = lifuren::audio::getAudioClient(client_name);
    if(!client) {
        SPDLOG_WARN("没有终端类型：{}", client_name);
        return;
    }
    const std::string& type = args[1];
    if(type == "train") {
        const std::string& model   = args[2];
        const std::string& dataset = args[3];
        lifuren::config::ModelParams params {
            .model_name = client_name,
            .check_path = lifuren::file::join({dataset, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
            .train_path = lifuren::file::join({dataset, lifuren::config::DATASET_TRAIN}).string(),
            .val_path   = lifuren::file::join({dataset, lifuren::config::DATASET_VAL  }).string(),
            .test_path  = lifuren::file::join({dataset, lifuren::config::DATASET_TEST }).string(),
        };
        client->trainValAndTest(params);
        client->save(model);
        SPDLOG_INFO("模型训练完成");
    } else if(type == "pred") {
        const std::string& model = args[2];
        const std::string& audio = args[3];
        client->load(model);
        const auto [success, output_file] = client->pred(audio);
        if(success) {
            SPDLOG_INFO("生成完成：{}", output_file);
        } else {
            SPDLOG_WARN("生成失败：{}", output_file);
        }
    } else {
        SPDLOG_WARN("无效类型：{}", type);
    }
}

// ./lifuren[.exe] image [chopin|mozart|wudaozi] [pred|train] model_file [image_file|dataset]
static void generateImage(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    const auto& client_name = args[0];
    auto client = lifuren::image::getImageClient(client_name);
    if(!client) {
        SPDLOG_WARN("没有终端类型：{}", client_name);
        return;
    }
    const std::string& type = args[1];
    if(type == "train") {
        const std::string& model   = args[2];
        const std::string& dataset = args[3];
        lifuren::config::ModelParams params {
            .model_name = client_name,
            .check_path = lifuren::file::join({dataset, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
            .train_path = lifuren::file::join({dataset, lifuren::config::DATASET_TRAIN}).string(),
            .val_path   = lifuren::file::join({dataset, lifuren::config::DATASET_VAL  }).string(),
            .test_path  = lifuren::file::join({dataset, lifuren::config::DATASET_TEST }).string(),
        };
        client->trainValAndTest(params);
        client->save(model);
        SPDLOG_INFO("模型训练完成");
    } else if(type == "pred") {
        const std::string& model = args[2];
        const std::string& image = args[3];
        client->load(model);
        const auto [success, output_file] = client->pred(image);
        if(success) {
            SPDLOG_INFO("生成完成：{}", output_file);
        } else {
            SPDLOG_WARN("生成失败：{}", output_file);
        }
    } else {
        SPDLOG_WARN("无效类型：{}", type);
    }
}

// ./lifuren[.exe] embedding dataset
static void embedding(const std::vector<std::string>& args) {
    if(args.size() < 1) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    const auto& type = args[0];
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    if(lifuren::audio::datasetPreprocessing(args[0])) {
        SPDLOG_INFO("音频嵌入成功");
    } else {
        SPDLOG_INFO("音频嵌入失败");
    }
}

// ./lifuren[.exe] embedding [audio|video] dataset
static void embeddingAudio(const std::vector<std::string>& args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    if(lifuren::audio::datasetPreprocessing(args[1])) {
        SPDLOG_INFO("音频嵌入成功");
    } else {
        SPDLOG_INFO("音频嵌入失败");
    }
}

// ./lifuren[.exe] [?|help]
static void help() {
    std::cout << R"(
./lifuren[.exe] 命令 [参数...]
./lifuren[.exe] audio [audio-shikuang] [pred|train] [model audio_file|dataset model_name]
./lifuren[.exe] video [video-wudaozi ] [pred|train] [model video_file|dataset model_name]
./lifuren[.exe] embedding [audio|video] dataset
./lifuren[.exe] [?|help]
)" << std::endl;
}

static void messageCallback(bool finish, const char* message) {
    #if defined(_DEBUG) || !defined(NDEBUG)
    // 测试时控制台日志已经打开
    #else
    std::cout << message << std::endl;
    #endif
}
