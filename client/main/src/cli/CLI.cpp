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

static void audio    (const std::vector<std::string>&); // 音频任务
static void image    (const std::vector<std::string>&); // 图片任务
static void embedding(const std::vector<std::string>&); // 数据嵌入
static void help(); // 帮助
static void messageCallback(const char*); // 消息回调

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
        ::audio(args);
    } else if(std::strcmp(command, "image") == 0) {
        ::image(args);
    } else if(std::strcmp(command, "embedding") == 0) {
        ::embedding(args);
    } else {
        SPDLOG_WARN("不支持的命令：{}", command);
        ::help();
    }
    lifuren::message::unregisterMessageCallback();
    return true;
}

static void audio(const std::vector<std::string>& args) {
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
        const std::string& model_path = args[2];
        const std::string& dataset    = args[3];
        lifuren::config::ModelParams params {
            .model_name = client_name,
            .model_path = model_path,
            .train_path = lifuren::file::join({dataset, lifuren::config::DATASET_TRAIN}).string(),
            .val_path   = lifuren::file::join({dataset, lifuren::config::DATASET_VAL  }).string(),
            .test_path  = lifuren::file::join({dataset, lifuren::config::DATASET_TEST }).string(),
        };
        client->trainValAndTest(params);
        client->save(lifuren::file::join({model_path, client_name + ".pt" }).string());
        SPDLOG_INFO("模型训练完成");
    } else if(type == "pred") {
        const std::string& model_file = args[2];
        const std::string& audio_file = args[3];
        client->load(model_file);
        const auto [success, output_file] = client->pred(audio_file);
        if(success) {
            SPDLOG_INFO("生成完成：{}", output_file);
        } else {
            SPDLOG_WARN("生成失败：{}", output_file);
        }
    } else {
        SPDLOG_WARN("无效类型：{}", type);
    }
}

static void image(const std::vector<std::string>& args) {
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
        const std::string& model_path = args[2];
        const std::string& dataset    = args[3];
        lifuren::config::ModelParams params {
            .model_name = client_name,
            .model_path = model_path,
            .train_path = lifuren::file::join({dataset, lifuren::config::DATASET_TRAIN}).string(),
            .val_path   = lifuren::file::join({dataset, lifuren::config::DATASET_VAL  }).string(),
            .test_path  = lifuren::file::join({dataset, lifuren::config::DATASET_TEST }).string(),
        };
        client->trainValAndTest(params);
        client->save(lifuren::file::join({model_path, client_name + ".pt"}).string());
        SPDLOG_INFO("模型训练完成");
    } else if(type == "pred") {
        const std::string& model_file = args[2];
        const std::string& image_file = args[3];
        client->load(model_file);
        const auto [success, output_file] = client->pred(image_file);
        if(success) {
            SPDLOG_INFO("生成完成：{}", output_file);
        } else {
            SPDLOG_WARN("生成失败：{}", output_file);
        }
    } else {
        SPDLOG_WARN("无效类型：{}", type);
    }
}

static void embedding(const std::vector<std::string>& args) {
    if(args.size() < 2) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    const auto& type    = args[0];
    const auto& dataset = args[1];
    if(type == "bach" && lifuren::audio::datasetPreprocessingBach(args[1])) {
        SPDLOG_INFO("音频嵌入成功");
    } else if(type == "shikuang" && lifuren::audio::datasetPreprocessingShikuang(args[1])) {
        SPDLOG_INFO("音频嵌入成功");
    } else {
        SPDLOG_INFO("音频嵌入失败");
    }
}

static void help() {
    std::cout << R"(
./lifuren[.exe] 命令 [参数...]
./lifuren[.exe] audio [bach|shikuang|beethoven] [pred|train] [model_file|model_path] [audio_file|xml_file|dataset]
./lifuren[.exe] image [chopin|mozart|wudaozi]   [pred|train] [model_file|model_path] [image_file|dataset]
./lifuren[.exe] embedding [bach|shikuang] dataset
./lifuren[.exe] [?|help]
)" << std::endl;
}

static void messageCallback(const char* message) {
    #if defined(_DEBUG) || !defined(NDEBUG)
    // 测试时控制台日志已经打开
    #else
    std::cout << message << std::endl;
    #endif
}
